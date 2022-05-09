import os
import tempfile

import torch
import cog
from cog import BasePredictor, Input
from torch import inference_mode
from torch.cuda.amp import autocast

from jaxformer.hf.codegen.modeling_codegen import CodeGenForCausalLM
from jaxformer.hf.sample import (create_custom_gpt2_tokenizer, print_time,
                                 sample, set_env, set_seed, truncate)

# This is a prompt engineering trick I learned from copilot. 
# Setting the shebang can be good for getting the "parsing args" modality to work. (anecdotal evidence)
SHEBANG = """
#!/usr/bin/env python3
"""

# Here, we hardcode a bit of example usage of the cog api. 
# This primes the model with context for how cog works, without needing to be overly specific about _all_ possible programs. 
# It's important to pack a lot of use cases into these examples to give the model more to go off of.
# Since plain english is understood by the model (to a degree), you can inform it what to do in comments in the code.
EXAMPLE_OF_COG_STYLE_PROGRAM = f"""
# cog requires you to override the `cog.BasePredictor` class method `predict`.
# cog makes use of type hints to infer the type of the input and output.
# e.g.
# Example 1:
# def predict(self, init_image: cog.Path = cog.Input(description="Path to image to predict on.")) -> cog.Path:
#    prediction = self.model(init_image)
#    torchvision.transforms.ToPILImage()(prediction).save("prediction.png")
#    return cog.Path("prediction.png")
# Example 2:
# class Predictor(cog.BasePredictor):
#     def setup(self):
#         self.model = self.load_model()
#     def predict(
#         self,
#         seed: int = cog.Input(description="Random seed. -1 for random", ge=-1, le=2147483647, default=-1),
#         batch_size: int = cog.Input(description="Batch size. Number of generations to predict", ge=1, le=8, default=1),
#         iterations: str = cog.Input(description="Number of iterations to predict", ge=1, le=100, default=1),
#         temperature: float = cog.Input(description="Temperature to use", ge=0.0, le=1.0, default=0.8),
#         top_k: int = cog.Input(description="Number of top tokens to return", ge=1, le=100, default=1),
#         top_p: float = cog.Input(description="Number of top probability to return", ge=0.0, le=1.0, default=0.9),
#         no_repeat_ngrams: int = cog.Input(description="Number of ngrams to avoid repeating", ge=0, le=100, default=0),
#         no_line_breaks: bool = cog.Input(description="Whether to avoid line breaks", default=False),
#         no_word_ngrams: bool = cog.Input(description="Whether to avoid word ngrams", default=False),
#         **kwargs
#     ):
"""

# This is the code we want to convert to a cog-style program. We assume the somewhat common case that they have an existing script that uses argparse.
ARGPARSE_CODE_TO_CONVERT = """
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--init_image', type=str, required = False, default = None, help='init image to use')
    parser.add_argument('--skip_timesteps', type=int, required = False, default = 0, help='how many diffusion steps are gonna be skipped')
    parser.add_argument('--prefix', type = str, required = False, default = '', help='prefix for output files')
    parser.add_argument('--num_batches', type = int, default = 1, required = False, help='number of batches')
    parser.add_argument('--batch_size', type = int, default = 1, required = False, help='batch size')
    parser.add_argument('--width', type = int, default = 256, required = False, help='image size of output (multiple of 8)')
    parser.add_argument('--height', type = int, default = 256, required = False, help='image size of output (multiple of 8)')
    parser.add_argument('--seed', type = int, default=-1, required = False, help='random seed')
    parser.add_argument('--guidance_scale', type = float, default = 5.0, required = False, help='classifier-free guidance scale')
    parser.add_argument('--steps', type = int, default = 0, required = False, help='number of diffusion steps')
    parser.add_argument('--cutn', type = int, default = 16, required = False, help='Number of cuts')
    return parser.parse_args()
"""
# I prepend each line with a comment to make sure it's interpreted as a comment. We don't want this to show in the final output however.
ARGPARSE_CODE_TO_CONVERT_COMMENTED_OUT = ["# " + line for line in ARGPARSE_CODE_TO_CONVERT.split("\n")]

# Here we explicitly ask for CodeGen to generate the predict method in particular. 
# Since setup is undefined, we stub it out and call an imagined `self.model = self.load_model()` method so that CodeGen won't be distracted by it.
COG_PROMPT = """
import cog
class Predictor(cog.BasePredictor):
    def setup(self):
        self.model = self.load_model()

    # TODO - modify the argparse version to use the cog predict with `cog.Input` decorators instead of manually parsing the args
    def predict(self, 
"""

DEFAULT_CONTEXT = f"""
{SHEBANG}
{EXAMPLE_OF_COG_STYLE_PROGRAM}
# Previous code, run manually with `python sample.py ...`
{ARGPARSE_CODE_TO_CONVERT_COMMENTED_OUT}
{COG_PROMPT}
"""

class Predictor(BasePredictor):
    @inference_mode()
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.model_name = "codegen-350M-mono"
        ckpt = os.path.join("checkpoints", self.model_name)
        assert os.path.isdir(ckpt), f"Model directory {ckpt} does not exist"
        self.device = torch.device("cuda")
        with print_time("loading parameters"):
            self.model = CodeGenForCausalLM.from_pretrained(
                ckpt,
                revision="float16",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
            self.model.to(self.device)

        with print_time("loading tokenizer"):
            self.tokenizer = create_custom_gpt2_tokenizer()
            self.tokenizer.padding_side = "left"
            self.pad = 50256
            self.tokenizer.pad_token = self.pad

    @inference_mode()
    def predict(
        self,
        context: str = Input(
            description="Some starting python code. CodeGen will try to complete the code provided, up to the max_length.",
            default=DEFAULT_CONTEXT,
        ),
        rng_seed: int = Input(description="Random number generator seed", default=0),
        rng_deterministic: bool = Input(description="Deterministic RNG", default=True),
        top_p: float = Input(
            description="Top-p sampling probability.", ge=0, le=1, default=0.95
        ),
        temperature: float = Input(
            description="Temperature for sampling", ge=0, le=1, default=0.2
        ),
        max_length: int = Input(
            description="Maximum length of generated text", ge=0, le=1000, default=512,
        ),
    ) -> cog.Path:
        """Run a single prediction on the model"""
        set_env()
        set_seed(rng_seed, deterministic=rng_deterministic)
        with print_time("sampling"):
            completion = sample(
                device=self.device,
                model=self.model,
                tokenizer=self.tokenizer,
                context=context,
                pad_token_id=self.pad,
                num_return_sequences=1,
                temp=temperature,
                top_p=top_p,
                max_length_sample=max_length,
            )[0]
            truncation = truncate(completion)

        # cog handles markdown files with the `.md` extension, 
        # so we need to write the output to a file inside after wrapping in a md code block.
        formatted_output = f"# Prediction\n\n```python\n{COG_PROMPT}{truncation}\n```\n"
        with open(tempfile.mktemp(suffix=".md"), "w") as f:
            f.write(formatted_output)
        return cog.Path("prediction.md")