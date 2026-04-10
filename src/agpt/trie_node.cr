module MicroGPT
  module AGPT
    # A weighted prefix trie node.
    #
    # `next_token_counts[x]` records how often token `x` follows that prefix.
    class TrieNode
      alias ChildEntry = {Int32, TrieNode}
      alias CountEntry = {Int32, Int32}

      getter id : Int32
      getter parent : TrieNode?
      getter token_id : Int32?
      getter depth : Int32
      getter children : Array(ChildEntry)
      getter next_token_counts : Array(CountEntry)

      def initialize(
        @id : Int32,
        @parent : TrieNode? = nil,
        @token_id : Int32? = nil,
        @depth : Int32 = 0
      )
        @children = [] of ChildEntry
        @next_token_counts = [] of CountEntry
      end

      def total_outgoing_mass : Int32
        @next_token_counts.sum(0) { |(_, count)| count }
      end

      def terminal? : Bool
        @children.empty? && @next_token_counts.empty?
      end

      def observe(next_token : Int32)
        @next_token_counts.each_with_index do |(token_id, count), index|
          next unless token_id == next_token
          @next_token_counts[index] = {token_id, count + 1}
          return
        end
        @next_token_counts << {next_token, 1}
      end

      def child_for(token : Int32) : TrieNode?
        @children.each do |child_token, child|
          return child if child_token == token
        end
        nil
      end

      def ensure_child(token : Int32, next_id : Int32) : TrieNode
        existing = child_for(token)
        return existing if existing

        child = TrieNode.new(
          id: next_id,
          parent: self,
          token_id: token,
          depth: @depth + 1
        )
        @children << {token, child}
        child
      end

      def next_token_counts_hash : Hash(Int32, Int32)
        counts = {} of Int32 => Int32
        @next_token_counts.each do |token_id, count|
          counts[token_id] = count
        end
        counts
      end

      def replace_next_token_counts(entries : Array(CountEntry))
        @next_token_counts.clear
        @next_token_counts.concat(entries)
      end
    end
  end
end
