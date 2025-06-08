from src.finetune import finetune
from src.unlearn import unlearn

def main():
    # output_dir = finetune()
    output_dir = "finetuned_llama3_yelp"
    print(unlearn(output_dir))

if __name__ == "__main__":
    main()
