# Trie sparsity profile.
#
# Loads a radix trie (global-radix format: radix_depth_NNN.bin + meta.bin) and
# prints a depth-by-depth profile of:
#   - number of radix endpoint nodes at that depth,
#   - branching factor distribution (how many are singletons / low / high),
#   - count distribution (min/median/max of total count per endpoint),
#   - fraction of endpoints that are singletons (= deterministic next-token).
#
# Usage: bin/trie-profile <radix_dir>
#
# The tool only loads from a global-radix format (NOT per-subtree). For
# per-subtree formats, concatenate the files or write a thin multiplexer.

require "../agpt"

if ARGV.size < 1
  STDERR.puts "Usage: trie-profile <radix_dir>"
  exit 1
end

dir = ARGV[0]

# Detect per-subtree vs global radix format.
if File.exists?(File.join(dir, "manifest.bin"))
  STDERR.puts "Per-subtree format detected. This tool currently only handles global-radix (radix_depth_*.bin at top level)."
  STDERR.puts "Point it at a dir like /tmp/agpt_input_d16_radix instead of /tmp/agpt_input_d16_radix_pst."
  exit 1
end

puts "Loading radix trie: #{dir}"
reader = MicroGPT::AGPT::RadixTrieReader.new(dir, max_cached: 8)
puts "  radix_count: #{reader.radix_count}"
puts "  depth_file_count: #{reader.depth_file_count}"
puts "  total_edge_chars: #{reader.total_edge_chars}"
puts "  vocab_size: #{reader.vocab_size}"
puts "  corpus_token_count: #{reader.corpus_token_count}"
puts ""

puts "Endpoint-depth profile (per endpoint depth d):"
puts "  n_nodes      = radix endpoints with endpoint_depth == d"
puts "  total_count  = Σ counts_val across those endpoints (mass that hits"
puts "                 a branching decision at this depth; a D-gram may"
puts "                 contribute to multiple depths)"
puts "  avg_branch   = mean # of distinct next-tokens per endpoint (= counts.size)"
puts "                 interior depths ≥ 2 by construction; cap can be 1"
puts "  avg_edge_len = mean length of the unary-compressed edge ending at d"
puts "                 1 = no compression; larger = bigger unary chain absorbed"
puts "  chars_absorbed = Σ (edge_len - 1) = # of original (leveled-trie) nodes"
puts "                   collapsed INTO edges ending at this depth"
puts ""
puts "  depth  n_nodes    total_count  median_cnt  mean_cnt   max_cnt  avg_branch  avg_edge_len  chars_absorbed  %singleton(at cap only)"

(1..reader.depth_file_count - 1).each do |d|
  records = reader.nodes_at_endpoint_depth(d)
  next if records.empty?

  totals = records.map { |r| r.counts.sum { |pair| pair[1] } }
  branches = records.map { |r| r.counts.size }
  edge_lens = records.map { |r| r.edge_len }

  n = records.size
  total_count_all = totals.sum
  sorted = totals.sort
  median = sorted[n // 2]
  mean = total_count_all.to_f / n
  max = sorted.last
  avg_branch = branches.sum.to_f / n
  avg_edge_len = edge_lens.sum.to_f / n
  chars_absorbed = edge_lens.sum - n  # (edge_len - 1) summed across endpoints

  is_cap = (d == reader.depth_file_count - 1)
  singletons = records.count { |r| r.counts.size <= 1 }
  singleton_str = is_cap ? sprintf("%.2f%%", 100.0 * singletons / n) : "   (n/a)"

  printf "  %5d  %9d  %13d  %10d  %9.2f  %9d  %9.2f  %12.2f  %14d  %s\n",
    d, n, total_count_all, median, mean, max,
    avg_branch, avg_edge_len, chars_absorbed, singleton_str
end

# Summary totals
puts ""
total_radix_endpoints = (0..reader.depth_file_count - 1).sum { |d| reader.nodes_at_endpoint_depth(d).size }
total_edge_chars = reader.total_edge_chars
absorbed_chars_total = total_edge_chars - total_radix_endpoints
compression_ratio = total_edge_chars.to_f / total_radix_endpoints
printf "Totals:  radix endpoints=%d, total edge chars=%d, absorbed=%d (%.2f%% of leveled nodes), compression=%.2fx\n",
  total_radix_endpoints, total_edge_chars, absorbed_chars_total,
  100.0 * absorbed_chars_total / total_edge_chars, compression_ratio
