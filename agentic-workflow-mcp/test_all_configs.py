import os
import glob

from core.internal import initialize, process
from core.util import load_workflow_config
import asyncio

config_examples_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config_examples'))
config_files = glob.glob(os.path.join(config_examples_dir, '*.json'))
all_configs = config_files

test_prompts = {
    "sequential_config": "Provide information about Antarctica",
    "branches_config": "Provide information about Nvidia RTX 3090 sales",
    "evaluator_optimizer_config": "Translate Is this really a evaluator optimized workflow? to French",
    "explicit_edges_config": "to draft an email",
    "parallel_config": "Agenda is to build a CICD pipeline",
    "routers_config": "I would like to change my email",
    "orchestrator_worker_config": "to read the \".gitignore\" file",
    "complex_combined_config": (
        "Summarize the following news article: "
        'The "most intense global coral bleaching event ever" has so far struck 84 per cent of the world\'s reefs and is ongoing, '
        'the International Coral Reef Initiative (ICRI) — a global partnership between nations and non-governmental and international organizations focused on sustainable management of coral reefs — reported on Wednesday.\n\n'
        'The new figure is far worse than previous events that hit 21 to 68 per cent of reefs.\n\n'
        'But scientists say the reefs and the corals are not all dead yet and could still  bounce back if people take the right steps, including conservation and cutting greenhouse gas emissions.\n\n'
        'Corals are small marine animals that live in colonies with colorful symbiotic algae that give them their rainbow hues and supply them with most of their food. But when the water gets too warm for too long, the algae release toxic compounds, and the corals expel them, leaving behind a white skeleton — causing "bleaching."'
    ),
    "embedding_retrieval_config": (
        "1. to embed files #file:Readme.md\n\n"
        "2. to create an email about Agentic Workflow Server if it has custom embedding support"
    ),
    "embedding_updater_config": "for folder `config_examples`",
    "complex_evaluator_optimizer_config": "Process the workflow"
}

os.environ["WORKSPACE_PATH"] = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

async def main():
    idx = 0
    while idx < len(all_configs):
        config_file = all_configs[idx]
        config_name = os.path.splitext(os.path.basename(config_file))[0]
        next_config_name = None
        if idx + 1 < len(all_configs):
            next_config_name = os.path.splitext(os.path.basename(all_configs[idx + 1]))[0]
        prompt_msg = (
            f"Process config ({config_name})? (y/n/s for skip): "
        )
        while True:
            user_input = input(prompt_msg).strip().lower()
            if user_input == "y":
                break
            elif user_input == "n":
                print("Exiting.")
                return
            elif user_input == "s":
                print(f"Skipped config: {config_name}")
                idx += 1
                # Prompt for the next config (if any)
                break
            else:
                print("Please enter 'y', 'n', or 's'.")
        if user_input == "y":
            os.environ["WORKFLOW_CONFIG_PATH"] = config_file
            user_prompt = test_prompts.get(config_name, "Default test prompt")
            try:
                workflow_config = load_workflow_config(os.getenv("WORKFLOW_CONFIG_PATH"))
                print(f"Successfully loaded config: {config_file}")
                print(f"Using prompt: {user_prompt}")

                app_context = await initialize(workflow_config)
                response = await process(app_context, user_prompt=user_prompt)
                print(f"Successfully processed workflow: {response}")
            except Exception as e:
                print(f"Error: Failed to load config: {config_file} with error: {e}")
            idx += 1

if __name__ == "__main__":
    asyncio.run(main())
