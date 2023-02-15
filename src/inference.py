import argparse
from peft import PeftModel, PeftConfig
from queue import Queue
import threading
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList

# Queue for streaming output
queue = Queue()

# Our streaming hack using stopping criteria :)


class Stream(StoppingCriteria):
    def __init__(self, q):
        self.q = q

    def __call__(self, input_ids, scores) -> bool:
        # input_ids is a list of lists, where the inner list is the
        # token ids for a single sequence. We only have one sequence
        # in this case, so we take the first inner list.

        # Place the latest token from the single sequence in our queue
        self.q.put(input_ids[0][-1])
        return False


def gen(model, batch, max_new_tokens, temperature, top_p, repetition_penalty):
    model.generate(
        **batch,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        stopping_criteria=StoppingCriteriaList([Stream(queue)]),
    )
    queue.put(False)


def run_peft(args):

    # Get Peft config, and associated tokenizer
    config = PeftConfig.from_pretrained(args.model)
    tokenizer_model = args.tokenizer_model if args.tokenizer_model else config.base_model_name_or_path

    # Load the base model
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path)

    # Load our trained adapter
    print("Loading PEFT adapter...")
    model = PeftModel.from_pretrained(base_model, args.model)

    # Load our tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)

    # Check if we need to add a pad token, and if so, add it
    # and resize the model accordingly
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))

    print("Sending model to cuda...")
    model.cuda()
    model.eval()

    print("Entering generation loop. To exit, type 'exit' or 'quit'.")
    #
    wants_exit = False
    while not wants_exit:
        # Take input and check for exit
        prompt = input("Enter the start of a quote: ")
        if (prompt.lower() == "exit" or prompt.lower() == "quit"):
            wants_exit = True
            continue

        # Tokenize our prompt and generate with our fancy async loop
        # which lets us stream the output token by token :)
        batch = tokenizer(prompt, return_tensors='pt').to('cuda')
        with torch.cuda.amp.autocast():
            gen_kwargs = {
                "model": model,
                "batch": batch,
                "max_new_tokens": args.max_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "repetition_penalty": args.repetition_penalty
            }

            print('\n===== START =====\n')
            threading.Thread(target=gen, kwargs=gen_kwargs).start()
            sequence = []
            print(prompt, end='', flush=True)
            while True:
                i = queue.get()
                if (i is False):
                    break
                else:
                    # Decode the token, add it to our sequence, and print it
                    next_token = tokenizer.decode(i, skip_special_tokens=True)
                    sequence.append(next_token)
                    print(next_token, end='', flush=True)

        # Print a final few lines to signal that we're done
        print('\n===== DONE =====\n')


def extract_args(args):
    parser = argparse.ArgumentParser()

    # ===
    # Model url or path
    # ===
    parser.add_argument("--model", type=str, required=True)

    # ===
    # Allow a custom tokenizer model
    # ===
    parser.add_argument("--tokenizer_model", type=str,
                        required=False, default=None)

    model_config = parser.add_argument_group("Model configuration")
    model_config.add_argument("--max_tokens", type=int,
                              default=64, required=False)
    model_config.add_argument(
        "--temperature", type=float, default=0.9, required=False)
    model_config.add_argument("--top_p", type=float,
                              default=1.0, required=False)
    model_config.add_argument("--repetition_penalty",
                              type=float, default=10.0, required=False)

    # Parse the above arguments
    return parser.parse_args(args)


if __name__ == "__main__":
    import sys

    args = extract_args(sys.argv[1:])
    run_peft(args)
