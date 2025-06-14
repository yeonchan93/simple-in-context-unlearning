import argparse

from src.finetune import finetune
from src.unlearn import unlearn
from src.dataset import main_dataset_task

def main(args):
    if args.task == "dataset":
        main_dataset_task(args)
    elif args.task == "finetune":
        finetune(args)
    elif args.task == "eval":
        unlearn(args, args.model_dir + args.type)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default="paper", help="can be paper or project")
    parser.add_argument('--task', type=str, default="eval", help="can be dataset, finetune or eval")
    parser.add_argument('--output_dir', type=str, default="./assets/")
    parser.add_argument('--model_dir', type=str, default="finetuned_llama_")
    parser.add_argument('--dataset_size', type=int, default=25000)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--num_train_epochs', type=int, default=1)
    parser.add_argument('--n_ctxt', type=int, default=3, help="Number of context examples (including the forget point)")
    parser.add_argument('--label_flipping_method', type=str, default="first-k", choices=["first-k", "last-k"])
    parser.add_argument('--max_length', type=int, default=1024)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    main(args)
