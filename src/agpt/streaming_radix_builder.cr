module MicroGPT
  module AGPT
    # Builds a radix-compressed trie index from an existing leveled trie index.
    #
    # A radix node is a branching endpoint (or leaf) that owns an incoming edge
    # of L characters. Runs of unary nodes (nodes with entry_count==1 in the
    # leveled trie) are collapsed into a single edge.
    #
    # In a leveled trie node N:
    #   entry_count == 0 : no next-token observations (end-of-corpus leaf) → drop
    #   entry_count == 1 : unary, exactly one child → extend the edge
    #   entry_count >= 2 : branching → emit radix node
    #
    # Binary format (radix_depth_NNN.bin where NNN is ENDPOINT character depth):
    #   magic (u32 LEVELED_MAGIC)
    #   depth (i32)
    #   record_count (i32)
    #   per record:
    #     radix_id         (i32)
    #     parent_radix_id  (i32)
    #     first_char_depth (i32)
    #     edge_len         (i32) — L
    #     edge_tokens[L]   (i32 × L)
    #     entry_count      (i32)
    #     entries[]        : (token_id i32, count i32)
    class StreamingRadixBuilder
      RADIX_MAGIC   = 0x52445841_u32  # 'RDXA'
      # v2 adds edge_mass (sum of counts at the FIRST original-trie node in the edge),
      # used for corpus-mass-weighted training. Mass is preserved along pure unary
      # chains (no branching, no truncation), so the head count equals the true
      # prefix frequency — avoids truncation-reduced endpoint counts.
      RADIX_VERSION = 2_i32

      # per_subtree: when true, emit one file per root-child subtree (radix_subtree_NNNNNN.bin)
      # + a manifest.bin listing them. Enables per-subtree loading for memory scaling
      # at large depths (d=32+) where a global KV cache would exceed available memory.
      def initialize(
        @reader : LeveledTrieReader,
        @out_dir : String,
        @progress : Bool = true,
        @per_subtree : Bool = false
      )
      end

      def build : NamedTuple(radix_count: Int32, total_edge_chars: Int64, max_endpoint_depth: Int32)
        Dir.mkdir_p(@out_dir)

        build_started = Time.instant

        # Build per-depth children indexes on demand: depth d+1 → parent_id → [records]
        # We lazily populate and keep only the currently-needed ones (small footprint).
        children_by_depth = {} of Int32 => Hash(Int32, Array(LeveledTrieReader::LoadedRecord))

        # Frontier: edge-start points to explore.
        # Each entry: (parent_radix_id, starting_original_id, starting_char_depth)
        # starting_original_id is the PARENT node in the original trie; we iterate
        # its children_at_{starting_char_depth}.
        frontier = Deque({Int32, Int32, Int32}).new
        frontier << {0, 0, 1}   # from virtual root, children at depth 1

        # Record tuple: {radix_id, parent_radix_id, first_char_depth, edge_tokens, edge_mass, endpoint_counts}
        radix_records_by_endpoint = {} of Int32 => Array({Int32, Int32, Int32, Array(Int32), Int32, Array({Int32, Int32})})
        radix_records_by_subtree  = {} of Int32 => Array({Int32, Int32, Int32, Array(Int32), Int32, Array({Int32, Int32})})
        # Track each radix node's root_child ancestor so we know which subtree it belongs to.
        root_child_of = {} of Int32 => Int32
        next_radix_id = 1_i32  # 0 reserved for virtual root
        radix_count = 1        # count the virtual root
        total_edge_chars = 0_i64
        max_endpoint_depth = 0

        while !frontier.empty?
          parent_radix_id, start_original_id, start_char_depth = frontier.shift
          next if start_char_depth >= @reader.depth_file_count

          children_idx = children_index_for(children_by_depth, start_char_depth)
          children = children_idx[start_original_id]?
          next if children.nil?

          children.each do |child|
            edge = [child.token]
            current = child
            current_depth = start_char_depth
            # Edge mass = sum of counts at the FIRST node of the edge. In a pure
            # unary chain, mass is preserved through every intermediate position,
            # so the head count is the true prefix frequency (not a truncation-
            # reduced endpoint count).
            head_counts = @reader.counts_of(child.id)
            edge_mass = head_counts.sum(0) { |t| t[1] }

            # Extend while unary
            loop do
              cnts = @reader.counts_of(current.id)
              if cnts.size == 1 && current_depth + 1 < @reader.depth_file_count
                # Unary: find the single child
                next_children_idx = children_index_for(children_by_depth, current_depth + 1)
                next_children = next_children_idx[current.id]?
                break if next_children.nil? || next_children.size != 1
                current = next_children[0]
                current_depth += 1
                edge << current.token
              else
                break
              end
            end

            # At this point, `current` is the endpoint. Its counts decide action.
            endpoint_counts = @reader.counts_of(current.id)
            if endpoint_counts.empty?
              # End-of-corpus leaf: drop entirely (no training signal, no descendants).
              next
            end

            # Emit radix node
            radix_id = next_radix_id
            next_radix_id += 1
            radix_count += 1
            total_edge_chars += edge.size

            endpoint_depth = current_depth
            if endpoint_depth > max_endpoint_depth
              max_endpoint_depth = endpoint_depth
            end

            # Determine root_child for this new radix node. If its parent is the
            # virtual root (parent_radix_id == 0), it IS a root_child itself.
            # Otherwise inherit from parent.
            rc = parent_radix_id == 0 ? radix_id : root_child_of[parent_radix_id]
            root_child_of[radix_id] = rc

            record = {radix_id, parent_radix_id, start_char_depth, edge, edge_mass, endpoint_counts}

            list = radix_records_by_endpoint[endpoint_depth]?
            if list.nil?
              list = [] of {Int32, Int32, Int32, Array(Int32), Int32, Array({Int32, Int32})}
              radix_records_by_endpoint[endpoint_depth] = list
            end
            list << record

            if @per_subtree
              slist = radix_records_by_subtree[rc]?
              if slist.nil?
                slist = [] of {Int32, Int32, Int32, Array(Int32), Int32, Array({Int32, Int32})}
                radix_records_by_subtree[rc] = slist
              end
              slist << record
            end

            # Descend: queue each branching child as a new edge-start
            if endpoint_counts.size >= 2
              frontier << {radix_id, current.id, current_depth + 1}
            end
          end

          # Keep children indexes resident for the full build — they are small
          # (~40 bytes/record) and eviction thrashes with the reader's own LRU.

          if @progress && radix_count % 10_000 == 0
            elapsed = (Time.instant - build_started).total_seconds
            STDERR.puts "[radix] #{radix_count} radix nodes, frontier=#{frontier.size}, max_ep_depth=#{max_endpoint_depth}  (#{elapsed.round(1)}s)"
          end
        end

        # Write per-endpoint-depth files
        (0..max_endpoint_depth).each do |d|
          records = radix_records_by_endpoint[d]? || ([] of {Int32, Int32, Int32, Array(Int32), Int32, Array({Int32, Int32})})
          write_depth_file(d, records)
        end

        # Per-subtree output mode: one file per root-child, plus a manifest. This
        # is for scalable training at large depths — the trainer can load ONE
        # subtree at a time and scope its KV cache to just that subtree's
        # character positions, instead of needing a global KV cache that grows
        # with total corpus positions.
        if @per_subtree
          Dir.mkdir_p(File.join(@out_dir, "subtrees"))
          manifest = [] of {Int32, Int32, Int64, Int32}   # {root_child_id, n_nodes, total_edge_chars, max_endpoint_depth}
          radix_records_by_subtree.each do |rc, recs|
            # Sort by endpoint depth for BFS-order loading.
            recs.sort_by! { |r| r[2] + r[3].size - 1 }
            st_edge_chars = 0_i64
            st_max_ep = 0
            recs.each do |r|
              st_edge_chars += r[3].size.to_i64
              ep = r[2] + r[3].size - 1
              st_max_ep = ep if ep > st_max_ep
            end
            write_subtree_file(rc, recs, st_max_ep)
            manifest << {rc, recs.size, st_edge_chars, st_max_ep}
          end
          # Sort manifest by root_child id for deterministic order
          manifest.sort_by! { |m| m[0] }
          write_manifest(manifest)
          STDERR.puts "[radix] per-subtree: #{manifest.size} subtree files written"
        end

        write_meta(
          radix_count,
          max_endpoint_depth + 1,
          total_edge_chars,
          @reader.index_metadata.corpus_token_count,
          @reader.index_metadata.vocab_size,
          @reader.index_metadata.corpus_hash,
          @reader.index_metadata.tokenizer_tag
        )

        elapsed = (Time.instant - build_started).total_seconds
        STDERR.puts "[radix] done: #{radix_count} radix nodes, #{total_edge_chars} total edge chars, #{max_endpoint_depth + 1} endpoint depths, #{elapsed.round(2)}s"

        {radix_count: radix_count, total_edge_chars: total_edge_chars, max_endpoint_depth: max_endpoint_depth}
      end

      private def children_index_for(
        cache : Hash(Int32, Hash(Int32, Array(LeveledTrieReader::LoadedRecord))),
        depth : Int32
      ) : Hash(Int32, Array(LeveledTrieReader::LoadedRecord))
        if existing = cache[depth]?
          return existing
        end
        # Build the index
        idx = {} of Int32 => Array(LeveledTrieReader::LoadedRecord)
        @reader.nodes_at_depth(depth).each do |rec|
          arr = idx[rec.parent_id]?
          if arr
            arr << rec
          else
            idx[rec.parent_id] = [rec]
          end
        end
        cache[depth] = idx
        idx
      end

      private def write_depth_file(
        d : Int32,
        records : Array({Int32, Int32, Int32, Array(Int32), Int32, Array({Int32, Int32})})
      )
        path = File.join(@out_dir, "radix_depth_#{"%03d" % d}.bin")
        File.open(path, "wb") do |io|
          io.write_bytes(RADIX_MAGIC, IO::ByteFormat::LittleEndian)
          io.write_bytes(d.to_i32, IO::ByteFormat::LittleEndian)
          io.write_bytes(records.size.to_i32, IO::ByteFormat::LittleEndian)
          records.each do |(radix_id, parent_radix_id, first_char_depth, edge, edge_mass, entries)|
            io.write_bytes(radix_id.to_i32, IO::ByteFormat::LittleEndian)
            io.write_bytes(parent_radix_id.to_i32, IO::ByteFormat::LittleEndian)
            io.write_bytes(first_char_depth.to_i32, IO::ByteFormat::LittleEndian)
            io.write_bytes(edge.size.to_i32, IO::ByteFormat::LittleEndian)
            edge.each { |tok| io.write_bytes(tok.to_i32, IO::ByteFormat::LittleEndian) }
            io.write_bytes(edge_mass.to_i32, IO::ByteFormat::LittleEndian) # v2: prefix mass
            io.write_bytes(entries.size.to_i32, IO::ByteFormat::LittleEndian)
            entries.each do |(token_id, count)|
              io.write_bytes(token_id.to_i32, IO::ByteFormat::LittleEndian)
              io.write_bytes(count.to_i32, IO::ByteFormat::LittleEndian)
            end
          end
        end
      end

      # Per-subtree file: one self-contained file per root-child subtree.
      # Header:
      #   magic (u32 RADIX_MAGIC)
      #   version (i32) — matches radix format version
      #   root_child_id (i32)
      #   record_count (i32)
      #   total_edge_chars (i64)
      #   max_endpoint_depth (i32)
      # Records: same layout as per-endpoint-depth files (radix_id, parent, fcd, edge_len, edge_tokens[], edge_mass, entry_count, entries[]).
      private def write_subtree_file(
        root_child_id : Int32,
        records : Array({Int32, Int32, Int32, Array(Int32), Int32, Array({Int32, Int32})}),
        max_endpoint_depth : Int32
      )
        dir = File.join(@out_dir, "subtrees")
        path = File.join(dir, "radix_subtree_#{"%06d" % root_child_id}.bin")
        total_edge_chars = 0_i64
        records.each { |r| total_edge_chars += r[3].size.to_i64 }
        File.open(path, "wb") do |io|
          io.write_bytes(RADIX_MAGIC, IO::ByteFormat::LittleEndian)
          io.write_bytes(RADIX_VERSION, IO::ByteFormat::LittleEndian)
          io.write_bytes(root_child_id.to_i32, IO::ByteFormat::LittleEndian)
          io.write_bytes(records.size.to_i32, IO::ByteFormat::LittleEndian)
          io.write_bytes(total_edge_chars.to_i64, IO::ByteFormat::LittleEndian)
          io.write_bytes(max_endpoint_depth.to_i32, IO::ByteFormat::LittleEndian)
          records.each do |(radix_id, parent_radix_id, first_char_depth, edge, edge_mass, entries)|
            io.write_bytes(radix_id.to_i32, IO::ByteFormat::LittleEndian)
            io.write_bytes(parent_radix_id.to_i32, IO::ByteFormat::LittleEndian)
            io.write_bytes(first_char_depth.to_i32, IO::ByteFormat::LittleEndian)
            io.write_bytes(edge.size.to_i32, IO::ByteFormat::LittleEndian)
            edge.each { |tok| io.write_bytes(tok.to_i32, IO::ByteFormat::LittleEndian) }
            io.write_bytes(edge_mass.to_i32, IO::ByteFormat::LittleEndian)
            io.write_bytes(entries.size.to_i32, IO::ByteFormat::LittleEndian)
            entries.each do |(token_id, count)|
              io.write_bytes(token_id.to_i32, IO::ByteFormat::LittleEndian)
              io.write_bytes(count.to_i32, IO::ByteFormat::LittleEndian)
            end
          end
        end
      end

      # Manifest: ordered list of subtree files.
      #   magic, version, n_subtrees
      #   per entry: root_child_id (i32), n_nodes (i32), total_edge_chars (i64), max_endpoint_depth (i32)
      private def write_manifest(manifest : Array({Int32, Int32, Int64, Int32}))
        path = File.join(@out_dir, "manifest.bin")
        File.open(path, "wb") do |io|
          io.write_bytes(RADIX_MAGIC, IO::ByteFormat::LittleEndian)
          io.write_bytes(RADIX_VERSION, IO::ByteFormat::LittleEndian)
          io.write_bytes(manifest.size.to_i32, IO::ByteFormat::LittleEndian)
          manifest.each do |(rc, n, chars, max_ep)|
            io.write_bytes(rc.to_i32, IO::ByteFormat::LittleEndian)
            io.write_bytes(n.to_i32, IO::ByteFormat::LittleEndian)
            io.write_bytes(chars.to_i64, IO::ByteFormat::LittleEndian)
            io.write_bytes(max_ep.to_i32, IO::ByteFormat::LittleEndian)
          end
        end
      end

      private def write_meta(
        radix_count : Int32,
        depth_file_count : Int32,
        total_edge_chars : Int64,
        corpus_token_count : Int32,
        vocab_size : Int32,
        corpus_hash : UInt64,
        tokenizer_tag : String
      )
        path = File.join(@out_dir, "meta.bin")
        File.open(path, "wb") do |io|
          io.write_bytes(RADIX_MAGIC, IO::ByteFormat::LittleEndian)
          io.write_bytes(RADIX_VERSION, IO::ByteFormat::LittleEndian)
          io.write_bytes(radix_count.to_i32, IO::ByteFormat::LittleEndian)
          io.write_bytes(depth_file_count.to_i32, IO::ByteFormat::LittleEndian)
          io.write_bytes(total_edge_chars.to_i64, IO::ByteFormat::LittleEndian)
          io.write_bytes(corpus_token_count.to_i32, IO::ByteFormat::LittleEndian)
          io.write_bytes(vocab_size.to_i32, IO::ByteFormat::LittleEndian)
          io.write_bytes(corpus_hash, IO::ByteFormat::LittleEndian)
          tag_bytes = tokenizer_tag.to_slice
          io.write_bytes(tag_bytes.size.to_i32, IO::ByteFormat::LittleEndian)
          io.write tag_bytes
        end
      end
    end
  end
end
