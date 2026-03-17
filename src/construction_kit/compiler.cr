require "./graph"
require "./executable_graph"

# Compiler: translates a visual GraphData into an ExecutableGraph.
# Phase 2: recursively walks container children graphs, creating individual
# ExecutableNodes for each building block and wiring them according to the
# actual graph topology. The graph IS the model.

module ConstructionKit
  class Compiler
    getter vocab_size : Int32
    getter seq_len : Int32
    getter lr : Float64

    def initialize(@vocab_size : Int32, @seq_len : Int32 = 128, @lr : Float64 = 0.0003)
    end

    # Compile a pipeline-level graph into an executable graph.
    def compile(graph : GraphData) : ExecutableGraph
      eg = ExecutableGraph.new

      # Recursively compile all nodes, flattening containers
      compile_scope(graph, nil, graph, eg, "")

      eg.compute_topo_order!
      eg
    end

    # Compile all nodes in a scope (graph), resolving boundary ports to the parent.
    # parent_node: the container node in parent_graph (nil for top-level)
    # parent_graph: the graph containing parent_node (nil for top-level)
    # eg: the flat ExecutableGraph we're building
    # prefix: dotted path for node ID namespacing
    private def compile_scope(
      graph : GraphData,
      parent_node : Node?,
      parent_graph : GraphData,
      eg : ExecutableGraph,
      prefix : String
    )
      # First pass: compile each node
      graph.nodes.each do |node|
        node_id = make_id(prefix, node.id)

        if container_type?(node.type) && node.children
          # Recursive: compile the container's children
          compile_scope(node.children.not_nil!, node, graph, eg, node_id)
        else
          # Leaf node: create an ExecutableNode
          exec_node = create_exec_node(node_id, node)
          eg.add_node(exec_node) if exec_node
        end
      end

      # Second pass: wire edges, resolving boundary ports
      graph.edges.each do |edge|
        from_id = resolve_from(edge.from, graph, parent_node, parent_graph, prefix)
        to_id = resolve_to(edge.to, graph, parent_node, parent_graph, prefix)
        next unless from_id && to_id

        # If source is a skipped node (data pipeline), trace back to boundary
        from_ok = from_id[0] == "__boundary__" || eg.nodes.has_key?(from_id[0])
        to_ok = to_id[0] == "__boundary__" || eg.nodes.has_key?(to_id[0])

        if !from_ok && to_ok
          # Source was skipped — trace backward through skipped nodes to find boundary
          traced = trace_through_skipped(edge.from, graph, prefix)
          if traced
            eg.add_edge(traced[0], traced[1], to_id[0], to_id[1])
          end
          next
        end

        next unless from_ok && to_ok
        eg.add_edge(from_id[0], from_id[1], to_id[0], to_id[1])
      end
    end

    # Is this node type a container that should be recursed into?
    private def container_type?(type : String) : Bool
      type == "cooperative" || type == "transformer" || type == "counter"
    end

    # Create a dotted ID path
    private def make_id(prefix : String, node_id : Int32) : String
      prefix.empty? ? node_id.to_s : "#{prefix}.#{node_id}"
    end

    # Resolve the FROM side of an edge
    # Returns {node_id, port_id} or nil if unresolvable
    private def resolve_from(
      port : PortRef,
      current_graph : GraphData,
      parent_node : Node?,
      parent_graph : GraphData,
      prefix : String
    ) : {String, String}?
      if port.nodeId == -1
        # Boundary output port: this scope's input comes from the parent graph.
        # Find the edge in parent_graph that connects TO parent_node on this port.
        return {"__boundary__", port.portId} unless parent_node

        parent_graph.edges.each do |pe|
          if pe.to.nodeId == parent_node.id && pe.to.portId == port.portId
            if pe.from.nodeId == -1
              # Grandparent boundary — pass through
              return {"__boundary__", pe.from.portId}
            else
              # Find the actual node in parent_graph. If it was compiled into
              # the flat graph, use its flattened ID.
              parent_prefix = prefix.rchop(".#{parent_node.id}")
              from_node = parent_graph.find_node(pe.from.nodeId)
              if from_node && container_type?(from_node.type) && from_node.children
                # The source is a container that was flattened — find its internal
                # output port. Look for boundary output edges in its children.
                from_prefix = make_id(parent_prefix, pe.from.nodeId)
                output_node = find_boundary_output_node(from_node.children.not_nil!, pe.from.portId, from_prefix)
                return output_node if output_node
              end
              return {make_id(parent_prefix, pe.from.nodeId), pe.from.portId}
            end
          end
        end
        nil
      else
        node = current_graph.find_node(port.nodeId)
        if node && container_type?(node.type) && node.children
          # Source is a flattened container — find its internal output node
          child_prefix = make_id(prefix, port.nodeId)
          output_node = find_boundary_output_node(node.children.not_nil!, port.portId, child_prefix)
          return output_node if output_node
        end
        {make_id(prefix, port.nodeId), port.portId}
      end
    end

    # Resolve the TO side of an edge
    private def resolve_to(
      port : PortRef,
      current_graph : GraphData,
      parent_node : Node?,
      parent_graph : GraphData,
      prefix : String
    ) : {String, String}?
      if port.nodeId == -1
        # Boundary input port: this scope's output goes to the parent graph.
        # Find the edge in parent_graph that connects FROM parent_node on this port.
        return {"__boundary__", port.portId} unless parent_node

        parent_graph.edges.each do |pe|
          if pe.from.nodeId == parent_node.id && pe.from.portId == port.portId
            if pe.to.nodeId == -1
              return {"__boundary__", pe.to.portId}
            else
              parent_prefix = prefix.rchop(".#{parent_node.id}")
              to_node = parent_graph.find_node(pe.to.nodeId)
              if to_node && container_type?(to_node.type) && to_node.children
                child_prefix = make_id(parent_prefix, pe.to.nodeId)
                input_node = find_boundary_input_node(to_node.children.not_nil!, pe.to.portId, child_prefix)
                return input_node if input_node
              end
              return {make_id(parent_prefix, pe.to.nodeId), pe.to.portId}
            end
          end
        end
        nil
      else
        node = current_graph.find_node(port.nodeId)
        if node && container_type?(node.type) && node.children
          child_prefix = make_id(prefix, port.nodeId)
          input_node = find_boundary_input_node(node.children.not_nil!, port.portId, child_prefix)
          return input_node if input_node
        end
        {make_id(prefix, port.nodeId), port.portId}
      end
    end

    # Trace backward through skipped (data pipeline) nodes to find the boundary.
    # e.g., windower:target_ids ← tokenizer:token_ids ← boundary:raw_text
    # For the graph executor, these pass-throughs become direct boundary connections.
    private def trace_through_skipped(port : PortRef, graph : GraphData, prefix : String) : {String, String}?
      node = graph.find_node(port.nodeId)
      return nil unless node

      # This node was skipped — find what feeds into it
      graph.edges.each do |e|
        next unless e.to.nodeId == node.id
        if e.from.nodeId == -1
          # Found the boundary — use the output port name from the skipped node
          return {"__boundary__", port.portId}
        else
          # Another node feeds this one — recurse
          result = trace_through_skipped(e.from, graph, prefix)
          return {"__boundary__", port.portId} if result
        end
      end
      nil
    end

    # Find the internal node that produces output for a container's boundary output port
    private def find_boundary_output_node(children : GraphData, port_id : String, prefix : String) : {String, String}?
      children.edges.each do |e|
        if e.to.nodeId == -1 && e.to.portId == port_id
          return {make_id(prefix, e.from.nodeId), e.from.portId}
        end
      end
      nil
    end

    # Find the internal node that receives input from a container's boundary input port
    private def find_boundary_input_node(children : GraphData, port_id : String, prefix : String) : {String, String}?
      children.edges.each do |e|
        if e.from.nodeId == -1 && e.from.portId == port_id
          return {make_id(prefix, e.to.nodeId), e.to.portId}
        end
      end
      nil
    end

    # Create an ExecutableNode for a leaf (non-container) node
    private def create_exec_node(id : String, node : Node) : ExecutableNode?
      case node.type
      when "embedding"
        d_model = node.param_i("d_model", 64)
        EmbeddingExec.new(id, @vocab_size, d_model, @seq_len)

      when "layer_norm"
        dim = node.param_i("dim", 64)
        LayerNormExec.new(id, dim)

      when "attention_layer"
        d_model = node.param_i("d_model", 64)
        n_heads = node.param_i("n_heads", 4)
        AttentionExec.new(id, d_model, n_heads, @seq_len)

      when "ffn_layer"
        d_model = node.param_i("d_model", 64)
        d_ff = node.param_i("d_ff", 256)
        FFNExec.new(id, d_model, d_ff)

      when "output_head"
        d_model = node.param_i("d_model", 64)
        OutputHeadExec.new(id, d_model, @vocab_size)

      when "loss"
        LossExec.new(id)

      when "stream_proj_internal"
        d_in = node.param_i("d_in", 64)
        d_out = node.param_i("d_out", 64)
        StreamProjExec.new(id, d_in, d_out)

      when "global_router"
        epsilon = node.param_f("epsilon", 0.2)
        # n_experts determined by incoming logits edges — default 2
        n_experts = 2  # TODO: infer from graph
        stream_dim = node.param_i("stream_dim", 64)
        GlobalRouterExec.new(id, n_experts, stream_dim, epsilon)

      # Data pipeline nodes — handled externally by dataset/builder
      when "char_tokenizer", "bpe_tokenizer",
           "sequential_window", "sliding_window", "random_window",
           "zero_init", "random_init", "learned_init",
           "optimizer", "stream_projector",
           "context_router", "gated_router"
        nil

      else
        nil  # Unknown type
      end
    end
  end
end
