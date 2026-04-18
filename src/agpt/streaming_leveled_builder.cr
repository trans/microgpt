module MicroGPT
  module AGPT
    # Builds the leveled trie index (same binary format as TrieCorpus#save_by_depth)
    # from the full corpus without holding the entire trie in RAM.
    #
    # Memory profile:
    #   active : Array({cursor, parent_node_id})  — at most corpus.size−1 entries, 8 bytes each
    #   groups hash (transient per depth)         — ~16 MB peak for large corpora
    #   depth_records (transient per depth)       — proportional to unique nodes at that depth
    #
    # For Shakespeare (1.1M positions, max_depth=128) peak RSS stays well under 100 MB.
    #
    # child_count and first_child fields are written as 0 / -1 (placeholder).
    # These fields are not used by LeveledTrieWalkTrainer or LeveledTrieReader's
    # training path, so they are safe to leave unpopulated.
    class StreamingLeveledBuilder
      LEVELED_MAGIC   = 0x4C475041_u32
      LEVELED_VERSION = 1_i32

      def initialize(
        @corpus : Array(Int32),
        @dir : String,
        @max_depth : Int32 = 128,
        @vocab_size : Int32 = 0,
        @corpus_hash : UInt64 = 0_u64,
        @tokenizer_tag : String = "unknown",
        @progress : Bool = true
      )
      end

      def build : Int32
        Dir.mkdir_p(@dir)

        n = @corpus.size
        # Each position 0..n-2 is a valid start (we need at least one next token).
        starts_used = n - 1

        # active = [(cursor_pos, parent_node_id)]
        # All starts begin at root (node 0).
        active = Array({Int32, Int32}).new(starts_used) { |i| {i, 0} }

        node_id_counter = 1   # 0 reserved for root
        total_nodes     = 1   # count root
        max_actual_depth = 0

        # depth_000.bin: root only
        write_depth_file(0, [{0, -1, -1, 0, [] of {Int32, Int32}}])

        build_started = Time.instant

        d = 1
        while !active.empty? && d <= @max_depth
          # Group active entries by (parent_id, corpus[cursor]).
          # Each group becomes one node at depth d.
          groups = {} of {Int32, Int32} => Array(Int32)
          active.each do |(cursor, parent_id)|
            tok = @corpus[cursor]
            key = {parent_id, tok}
            arr = groups[key]?
            if arr
              arr << cursor
            else
              groups[key] = [cursor]
            end
          end

          # Sort for deterministic node ordering: by (parent_id, token).
          sorted_keys = groups.keys.sort_by { |k| {k[0], k[1]} }

          next_active = [] of {Int32, Int32}
          # Preallocate records array with known size
          depth_records = Array({Int32, Int32, Int32, Int32, Array({Int32, Int32})}).new(sorted_keys.size)

          sorted_keys.each do |key|
            parent_id = key[0]
            tok       = key[1]
            cursors   = groups[key]

            nid = node_id_counter
            node_id_counter += 1
            total_nodes += 1

            # Build next-token distribution and collect next cursors.
            next_tok_counts = {} of Int32 => Int32
            cursors.each do |cursor|
              next_pos = cursor + 1
              if next_pos < n
                next_tok = @corpus[next_pos]
                cnt = next_tok_counts[next_tok]?
                next_tok_counts[next_tok] = (cnt || 0) + 1
                next_active << {next_pos, nid}
              end
            end

            entries = next_tok_counts.to_a.sort_by(&.first)
            # {id, parent_id, token, depth, counts}
            depth_records << {nid, parent_id, tok, d, entries}
          end

          write_depth_file(d, depth_records)
          max_actual_depth = d
          active = next_active

          if @progress
            elapsed = (Time.instant - build_started).total_seconds
            STDERR.puts "[agpt stream] depth #{"%3d" % d}: #{"%7d" % depth_records.size} nodes, #{"%8d" % active.size} active  (#{elapsed.round(1)}s)"
          end

          d += 1
        end

        depth_file_count = max_actual_depth + 1  # depths 0..max_actual_depth
        write_meta(total_nodes, depth_file_count, starts_used)

        elapsed = (Time.instant - build_started).total_seconds
        STDERR.puts "[agpt stream] done: #{total_nodes} nodes, #{depth_file_count} depth files, #{elapsed.round(1)}s total"
        total_nodes
      end

      private def write_depth_file(
        d : Int32,
        records : Array({Int32, Int32, Int32, Int32, Array({Int32, Int32})})
      )
        path = File.join(@dir, "depth_#{"%03d" % d}.bin")
        File.open(path, "wb") do |io|
          io.write_bytes(LEVELED_MAGIC, IO::ByteFormat::LittleEndian)
          io.write_bytes(d.to_i32, IO::ByteFormat::LittleEndian)
          io.write_bytes(records.size.to_i32, IO::ByteFormat::LittleEndian)
          records.each do |(nid, parent_id, tok, depth, entries)|
            io.write_bytes(nid.to_i32, IO::ByteFormat::LittleEndian)
            io.write_bytes(parent_id.to_i32, IO::ByteFormat::LittleEndian)
            io.write_bytes(tok.to_i32, IO::ByteFormat::LittleEndian)
            io.write_bytes(depth.to_i32, IO::ByteFormat::LittleEndian)
            io.write_bytes(0_i32, IO::ByteFormat::LittleEndian)   # child_count: placeholder
            io.write_bytes(-1_i32, IO::ByteFormat::LittleEndian)  # first_child: placeholder
            io.write_bytes(entries.size.to_i32, IO::ByteFormat::LittleEndian)
            entries.each do |(token_id, count)|
              io.write_bytes(token_id.to_i32, IO::ByteFormat::LittleEndian)
              io.write_bytes(count.to_i32, IO::ByteFormat::LittleEndian)
            end
          end
        end
      end

      private def write_meta(node_count : Int32, depth_file_count : Int32, starts_used : Int32)
        path = File.join(@dir, "meta.bin")
        File.open(path, "wb") do |io|
          io.write_bytes(LEVELED_MAGIC, IO::ByteFormat::LittleEndian)
          io.write_bytes(LEVELED_VERSION, IO::ByteFormat::LittleEndian)
          io.write_bytes(@max_depth.to_i32, IO::ByteFormat::LittleEndian)
          io.write_bytes(-1_i32, IO::ByteFormat::LittleEndian)  # max_starts: -1 = no limit
          io.write_bytes(0_i32, IO::ByteFormat::LittleEndian)   # start_offset
          io.write_bytes(starts_used.to_i32, IO::ByteFormat::LittleEndian)
          io.write_bytes(node_count.to_i32, IO::ByteFormat::LittleEndian)
          io.write_bytes(depth_file_count.to_i32, IO::ByteFormat::LittleEndian)
          io.write_bytes(@corpus.size.to_i32, IO::ByteFormat::LittleEndian)
          io.write_bytes(@vocab_size.to_i32, IO::ByteFormat::LittleEndian)
          io.write_bytes(@corpus_hash, IO::ByteFormat::LittleEndian)
          tag_bytes = @tokenizer_tag.to_slice
          io.write_bytes(tag_bytes.size.to_i32, IO::ByteFormat::LittleEndian)
          io.write(tag_bytes)
        end
      end
    end
  end
end
