from fairseq.moe_checkpoint_utils import initialize_moe_from_opt
import sys


if __name__ == '__main__':
    output_dir = sys.argv[1]
    num_experts = sys.argv[2]
    initialize_moe_from_opt(output_dir, num_experts)
