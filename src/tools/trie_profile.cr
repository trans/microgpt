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

puts "Endpoint-depth profile:"
puts "  depth  n_nodes       total_count  median_count  mean_count    max_count   %singletons  %multi  avg_branch_factor"

(1..reader.depth_file_count - 1).each do |d|
  records = reader.nodes_at_endpoint_depth(d)
  next if records.empty?

  # Per-record: total count = sum of counts_val
  totals = records.map do |r|
    r.counts.sum { |pair| pair[1] }
  end
  branches = records.map { |r| r.counts.size }

  n = records.size
  total_count_all = totals.sum
  sorted = totals.sort
  median = sorted[n // 2]
  mean = total_count_all.to_f / n
  max = sorted.last
  singletons = records.count { |r| r.counts.size <= 1 }
  multi = n - singletons
  avg_branch = branches.sum.to_f / n

  printf "  %5d  %9d  %13d  %12d  %10.2f  %10d  %10.2f%%  %5.2f%%  %.2f\n",
    d, n, total_count_all, median, mean, max,
    100.0 * singletons / n,
    100.0 * multi / n,
    avg_branch
end
