module MicroGPT
  module AGPT
    # Weighted next-token loss for one trie prefix node.
    #
    # Given logits for a single prefix state and observed outgoing token counts,
    # computes a mean weighted cross-entropy loss and a logits gradient suitable
    # for correctness-first AGPT training.
    class WeightedNextTokenLoss
      def loss_and_backward(logits : MicroGPT::Mat, counts : Hash(Int32, Int32)) : {Float64, MicroGPT::Mat}
        raise "WeightedNextTokenLoss expects a single-row logits matrix" unless logits.rows == 1
        raise "WeightedNextTokenLoss requires at least one observed next token" if counts.empty?

        probs = MicroGPT.backend.softmax_rows(logits)
        total = counts.values.sum(0)
        loss = 0.0

        counts.each do |token_id, count|
          loss -= count * Math.log(probs[0, token_id] + 1e-10)
        end
        loss /= total

        grad = MicroGPT::Mat.new(logits.rows, logits.cols)
        logits.cols.times do |j|
          grad[0, j] = probs[0, j]
        end
        counts.each do |token_id, count|
          grad[0, token_id] -= count.to_f32 / total
        end

        {loss, grad}
      end
    end
  end
end
