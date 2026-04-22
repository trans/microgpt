# Bayesian posterior density check for the Bayesian Bloom proposal.
#
# For a prefix trie at depth D, pick a target (prev, π)-length d_star (≤ D).
# For every radix node whose edge covers depth d_star, extract its
# d_star-char root-path and split as (prev, π) = (first_char, remaining).
# Bucket by π, then compute:
#   - |support|   : number of distinct prev tokens that precede π
#   - entropy(P)  : Shannon entropy of P(prev | π) in nats
#   - effective-V : exp(entropy)  (how many prevs effectively contribute)
#
# Shows the distribution across all π's, especially at sparse tips (forward
# branching factor = 1). If the backward posterior is near-uniform there,
# the bloom carries no information. If it's sharply peaked, it does.
#
# Usage: bin/bayesian-posterior <radix_dir> <d_star>

require "../agpt"

if ARGV.size < 2
  STDERR.puts "Usage: bayesian-posterior <radix_dir> <d_star>"
  STDERR.puts "  radix_dir: global-radix format (radix_depth_*.bin at top level)"
  STDERR.puts "  d_star:    length of (prev, π) path (prev = 1 char, π = d_star-1 chars)"
  exit 1
end

dir = ARGV[0]
d_star = ARGV[1].to_i

if File.exists?(File.join(dir, "manifest.bin"))
  STDERR.puts "Per-subtree format detected. Use global-radix (e.g. /tmp/agpt_input_d16_radix)."
  exit 1
end

puts "Loading radix trie: #{dir}  d_star=#{d_star}"
reader = MicroGPT::AGPT::RadixTrieReader.new(dir, max_cached: d_star + 2)

# Build id → record map by enumerating all depths up to d_star.
id_to_record = {} of Int32 => MicroGPT::AGPT::RadixTrieReader::LoadedRecord
(0..(reader.depth_file_count - 1)).each do |d|
  reader.nodes_at_endpoint_depth(d).each { |r| id_to_record[r.id] = r }
end
puts "  Loaded #{id_to_record.size} radix records"

# Compute root path for every record via memoized parent walk.
path_cache = {} of Int32 => Array(Int32)
path_cache[0] = [] of Int32

get_path = ->(rid : Int32) : Array(Int32) {
  stack = [] of Int32
  cur = rid
  while !path_cache.has_key?(cur)
    stack << cur
    cur = id_to_record[cur].parent_id
  end
  while !stack.empty?
    cid = stack.pop
    parent = id_to_record[cid].parent_id
    edge = id_to_record[cid].edge_tokens
    path_cache[cid] = path_cache[parent] + edge
  end
  path_cache[rid]
}

# Bucket: π (as Array(Int32)) → Hash(prev_tok → count).
# Only consider nodes whose edge covers depth d_star
# (first_char_depth ≤ d_star ≤ endpoint_depth).
buckets = {} of Array(Int32) => Hash(Int32, Int64)
nodes_scanned = 0
nodes_covering = 0

id_to_record.each_value do |r|
  next if r.id == 0
  fcd = r.first_char_depth
  ed  = fcd + r.edge_tokens.size - 1
  next unless fcd <= d_star && d_star <= ed
  nodes_scanned += 1
  nodes_covering += 1

  # Full root path through this node's edge.
  parent_path = path_cache.has_key?(r.parent_id) ? path_cache[r.parent_id] : get_path.call(r.parent_id)
  offset = d_star - fcd  # index into edge at depth d_star
  # Depth-d_star prefix:
  prefix = parent_path.dup
  (0..offset).each { |i| prefix << r.edge_tokens[i] }
  next unless prefix.size == d_star

  prev_tok = prefix[0]
  pi       = prefix[1..]
  bucket = buckets[pi] ||= {} of Int32 => Int64
  bucket[prev_tok] = (bucket[prev_tok]? || 0_i64) + r.edge_mass.to_i64
end

puts "  Nodes whose edge covers depth #{d_star}: #{nodes_covering}"
puts "  Unique π buckets: #{buckets.size}"
puts ""

# For each bucket compute support, entropy, and count(π) (total corpus occurrences).
if buckets.empty?
  puts "No buckets at d_star=#{d_star}. Try a smaller value."
  exit 0
end

support_sizes = [] of Int32
entropies = [] of Float64
effective_sizes = [] of Float64
pi_counts = [] of Int64  # total occurrences of π (sum over prevs)

buckets.each do |_pi, prev_counts|
  support_sizes << prev_counts.size
  total = 0_i64
  prev_counts.each_value { |c| total += c }
  pi_counts << total
  h = 0.0
  prev_counts.each_value do |c|
    p = c.to_f / total.to_f
    h -= p * Math.log(p) if p > 0.0
  end
  entropies << h
  effective_sizes << Math.exp(h)
end

def percentile(arr : Array(Float64), p : Float64) : Float64
  sorted = arr.sort
  idx = (p * (sorted.size - 1)).to_i
  sorted[idx]
end

def percentile_i(arr : Array(Int32), p : Float64) : Int32
  sorted = arr.sort
  idx = (p * (sorted.size - 1)).to_i
  sorted[idx]
end

puts "=============================================================================="
puts "POSTERIOR P(prev | π) at d_star=#{d_star}  (π length = #{d_star - 1})"
puts "=============================================================================="
puts "Across #{buckets.size} unique π's (each bucket aggregates mass across prevs):"
puts ""
puts "Support size (distinct prevs preceding π):"
puts "  min=#{support_sizes.min}  p25=#{percentile_i(support_sizes, 0.25)}  median=#{percentile_i(support_sizes, 0.5)}  p75=#{percentile_i(support_sizes, 0.75)}  max=#{support_sizes.max}  mean=#{sprintf("%.2f", support_sizes.sum.to_f / support_sizes.size)}"
puts ""
puts "Entropy H(P(prev | π)) in nats (uniform-over-V upper bound = ln #{reader.vocab_size} = #{sprintf("%.3f", Math.log(reader.vocab_size))}):"
puts "  min=#{sprintf("%.3f", entropies.min)}  p25=#{sprintf("%.3f", percentile(entropies, 0.25))}  median=#{sprintf("%.3f", percentile(entropies, 0.5))}  p75=#{sprintf("%.3f", percentile(entropies, 0.75))}  max=#{sprintf("%.3f", entropies.max)}  mean=#{sprintf("%.3f", entropies.sum / entropies.size)}"
puts ""
puts "Effective support = exp(H):"
puts "  min=#{sprintf("%.2f", effective_sizes.min)}  p25=#{sprintf("%.2f", percentile(effective_sizes, 0.25))}  median=#{sprintf("%.2f", percentile(effective_sizes, 0.5))}  p75=#{sprintf("%.2f", percentile(effective_sizes, 0.75))}  max=#{sprintf("%.2f", effective_sizes.max)}  mean=#{sprintf("%.2f", effective_sizes.sum / effective_sizes.size)}"
puts ""

# Histogram of support sizes
puts "Support histogram (# of π buckets with k distinct prevs):"
hist = {} of Int32 => Int32
support_sizes.each { |s| hist[s] = (hist[s]? || 0) + 1 }
keys = hist.keys.sort
keys.each do |k|
  puts "  support=#{k}: #{hist[k]} buckets"
end
puts ""

# Singletons: π that has only ONE prev — these are "backward-unary."
n_singleton = support_sizes.count(1)
puts "Backward-unary π (support=1, i.e., exactly one prev precedes π): #{n_singleton} / #{buckets.size} = #{sprintf("%.1f", n_singleton * 100.0 / buckets.size)}%"
puts ""
puts "=============================================================================="
puts "FILTERED: only π's that GENUINELY RECUR in the corpus (count(π) >= k)"
puts "=============================================================================="
puts "Removes π that appear just once — those are support=1 by arithmetic, not structure."
puts ""

[2_i64, 5_i64, 10_i64, 50_i64].each do |min_count|
  idxs = (0...buckets.size).select { |i| pi_counts[i] >= min_count }
  if idxs.empty?
    puts "count(π) >= #{min_count}: no π's qualify"
    next
  end
  filt_support = idxs.map { |i| support_sizes[i] }
  filt_entropy = idxs.map { |i| entropies[i] }
  filt_eff     = idxs.map { |i| effective_sizes[i] }
  n_filt = idxs.size
  n_unary_filt = filt_support.count(1)
  puts "count(π) >= #{min_count}: #{n_filt} π's  (#{sprintf("%.1f", n_filt * 100.0 / buckets.size)}% of total)"
  puts "  support         : min=#{filt_support.min}  median=#{percentile_i(filt_support, 0.5)}  mean=#{sprintf("%.2f", filt_support.sum.to_f / n_filt)}  max=#{filt_support.max}"
  puts "  entropy (nats)  : min=#{sprintf("%.3f", filt_entropy.min)}  median=#{sprintf("%.3f", percentile(filt_entropy, 0.5))}  mean=#{sprintf("%.3f", filt_entropy.sum / n_filt)}  max=#{sprintf("%.3f", filt_entropy.max)}"
  puts "  effective-V     : min=#{sprintf("%.2f", filt_eff.min)}  median=#{sprintf("%.2f", percentile(filt_eff, 0.5))}  mean=#{sprintf("%.2f", filt_eff.sum / n_filt)}  max=#{sprintf("%.2f", filt_eff.max)}"
  puts "  backward-unary  : #{n_unary_filt} / #{n_filt} = #{sprintf("%.1f", n_unary_filt * 100.0 / n_filt)}%"
  puts ""
end

puts "Interpretation (filtered view):"
puts "  For π's that genuinely recur (count(π) >= 2), how many distinct prevs typically precede them?"
puts "  If mean effective-V is still ≈ 1, the bloom carries no info (each recurring π always has the same prev)."
puts "  If mean effective-V is closer to V, the bloom is genuinely informative."
