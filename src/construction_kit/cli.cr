require "jargon"
require "./graph"
require "./builder"

module ConstructionKit
  SCHEMA = %({
    "type": "object",
    "positional": ["graph"],
    "properties": {
      "graph": {
        "type": "string",
        "description": "Path to a graph JSON file exported from the Construction Kit",
        "short": "g"
      },
      "data-file": {
        "type": "string",
        "description": "Training text file",
        "default": "data/input.txt",
        "short": "f"
      },
      "steps": {
        "type": "integer",
        "description": "Training steps to run before generation",
        "default": 0,
        "short": "s"
      },
      "seed": {
        "type": "string",
        "description": "Seed text for generation",
        "default": "First Citizen:\\n"
      },
      "max-tokens": {
        "type": "integer",
        "description": "Maximum number of tokens to generate",
        "default": 120
      },
      "temperature": {
        "type": "number",
        "description": "Sampling temperature",
        "default": 0.8
      },
      "strict": {
        "type": "boolean",
        "description": "Fail if graph validation reports errors",
        "default": false
      },
      "build-only": {
        "type": "boolean",
        "description": "Build the graph and print a summary without generating text",
        "default": false
      }
    },
    "required": ["graph"]
  })

  private def self.load_graph(path : String) : GraphData
    json = File.read(path)
    parsed = JSON.parse(json)
    if parsed["graph"]?
      GraphDocument.from_json(json).graph
    else
      GraphData.from_json(json)
    end
  end

  private def self.config_for(graph : GraphData, data_file : String) : ModelConfig
    extract_config(graph, data_file)
  rescue
    ModelConfig.new(
      data_file, 128, 64,
      [] of ExpertSpec,
      "none", 0.0,
      false, 0.0003
    )
  end

  private def self.print_summary(summary : ModelSummary)
    puts "Summary"
    puts "  params:     #{summary.total_params}"
    puts "  experts:    #{summary.n_experts}"
    puts "  router:     #{summary.router}"
    puts "  seq_len:    #{summary.seq_len}"
    puts "  stream_dim: #{summary.stream_dim}"
    puts "  vocab:      #{summary.vocab_size}"
    puts "  data_file:  #{summary.data_file}"
  end

  def self.main
    cli = Jargon.cli("construction-kit", json: SCHEMA)
    cli.run do |result|
      graph_path = result["graph"].as_s
      data_file = result["data-file"].as_s
      steps = result["steps"].as_i64.to_i
      seed = result["seed"].as_s
      max_tokens = result["max-tokens"].as_i64.to_i
      temperature = result["temperature"].as_f
      strict = result["strict"]?.try(&.as_bool) || false
      build_only = result["build-only"]?.try(&.as_bool) || false

      graph = load_graph(graph_path)
      errors = validate(graph)
      if errors.any?
        STDERR.puts "Validation"
        errors.each { |err| STDERR.puts "  - #{err}" }
        exit 1 if strict
      end

      builder = Builder.new(config_for(graph, data_file), graph, true)
      summary = builder.summary
      print_summary(summary)

      if (warnings = builder.exec_graph.try(&.warnings)) && !warnings.empty?
        STDERR.puts "Warnings"
        warnings.each { |warning| STDERR.puts "  - #{warning}" }
      end

      unless steps <= 0
        puts
        puts "Training"
        avg_loss = 0.0
        interval = Math.max(1, steps // 20)
        steps.times do |step|
          result = builder.train_step
          avg_loss = step == 0 ? result.loss : 0.99 * avg_loss + 0.01 * result.loss
          if step == 0 || (step + 1) % interval == 0 || step + 1 == steps
            puts "  step #{step + 1}/#{steps}  loss=#{"%.5f" % result.loss}  avg=#{"%.5f" % avg_loss}"
          end
        end
      end

      next if build_only

      puts
      puts "Generation"
      puts builder.generate(seed, max_tokens, temperature)
    end
  end
end

ConstructionKit.main
