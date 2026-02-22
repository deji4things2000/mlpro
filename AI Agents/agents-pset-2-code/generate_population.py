import math
import random
import logging
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
import asyncio
import hashlib
import json
import hydra
import litellm
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from pathlib import Path
from pydantic import ConfigDict
from pydantic import Field
from pydantic import create_model
from utils.parallel import to_thread_map


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    base_dir = Path(get_original_cwd())
    personas = asyncio.run(build_population(cfg, base_dir))
    out_path = base_dir / str(cfg.population.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"personas": personas}, indent=2))
    print(f"Wrote {len(personas)} personas to {out_path}")


def sigmoid(value: float) -> float:
    return 1 / (1 + math.exp(-value))


def stable_seed(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def validate_sampling(trait_names: list[str], sampling: dict) -> None:
    if str(sampling["method"]) != "explicit_per_trait_distribution":
        raise ValueError(f"Unsupported trait_sampling.method: {sampling['method']}")

    distributions = dict(sampling["distributions"])
    missing = [trait for trait in trait_names if trait not in distributions]

    if missing:
        raise ValueError(f"Missing trait distributions for: {missing}")

    for trait in trait_names:
        entry = dict(distributions[trait])
        values = list(entry["values"])
        probs = list(entry["probs"])

        if len(values) != len(probs):
            raise ValueError(f"Trait {trait} has mismatched values/probs lengths")

        total = sum(float(prob) for prob in probs)

        if abs(total - 1.0) > 1e-8:
            raise ValueError(f"Trait {trait} probabilities must sum to 1.0, got {total}")


def sample_traits(trait_names: list[str], sampling: dict, seed: int) -> dict[str, float]:
    rng = random.Random(seed)

    validate_sampling(trait_names, sampling)

    distributions = dict(sampling["distributions"])
    traits = {}

    for trait in trait_names:
        entry = dict(distributions[trait])
        values = list(entry["values"])
        probs = list(entry["probs"])
        traits[trait] = rng.choices(values, weights=probs, k=1)[0]

    return traits


def build_seed_profile(index: int, seed: int) -> str:
    rng = random.Random(seed + index)

    roles = [
        "computational biologist",
        "machine learning scientist",
        "experimental physicist",
        "cognitive science researcher",
        "applied statistician"
    ]
    domains = [
        "scientific software reliability",
        "shared compute governance",
        "reproducible experimentation",
        "collaborative model development",
        "resource-constrained research planning"
    ]
    stages = [
        "early-career",
        "mid-career",
        "senior"
    ]
    institutions = [
        "public research university",
        "private research institute",
        "interdisciplinary lab consortium"
    ]

    role = rng.choice(roles)
    domain = rng.choice(domains)
    stage = rng.choice(stages)
    institution = rng.choice(institutions)

    return (
        f"Persona-{index:03d} is an {stage} {role} at a {institution}. "
        f"They work on {domain} in a team-based lab setting."
    )


def expand_profile(seed_profile: str, model: str, rounds: int) -> str:
    profile = seed_profile
    for _ in range(rounds):
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "Expand a research persona with plausible concrete detail while preserving realism."
                },
                {
                    "role": "user",
                    "content": (
                        "Expand this profile by adding specific but realistic details about work habits, collaboration style, and typical constraints. "
                        "Keep it concise (4-6 sentences) and grounded.\n\n"
                        f"PROFILE:\n{profile}"
                    )
                }
            ]
        }
        response = litellm.completion(**payload)
        profile = response["choices"][0]["message"]["content"].strip()
    return profile


def generate_life_story(name: str, expanded_profile: str, model: str, target_traits: dict[str, float] | None = None) -> str:
    system_content = "Write concise third-person scientist life stories for behavioral simulation."
    user_content = (
        f"Name: {name}.\n"
        "Using the profile below, write a 170-230 word life story focusing on background, decision-making under uncertainty, and social behavior in teams.\n\n"
    )
    if target_traits:
        trait_str = ", ".join(f"{k}={v:.2f}" for k, v in target_traits.items())
        user_content += f"TARGET BIG-FIVE TRAITS (for conditioning behavior): {trait_str}\n\n"

    user_content += f"PROFILE:\n{expanded_profile}"

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": system_content
            },
            {
                "role": "user",
                "content": user_content
            }
        ]
    }
    response = litellm.completion(**payload)
    return response["choices"][0]["message"]["content"].strip()


def build_bfi_response_model(items: list[dict]):
    fields: dict[str, tuple[type[int], Field]] = {}
    for item in items:
        fields[str(item["id"])] = (int, Field(ge=1, le=5))
    return create_model("BFIResponse", __config__=ConfigDict(extra="forbid"), **fields)


def administer_bfi(life_story: str, items: list[dict], model: str) -> dict[str, int]:
    item_lines = [f"{item['id']}: {item['text']}" for item in items]
    response_model = build_bfi_response_model(items)
    payload = {
        "model": model,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "bfi_response",
                "strict": True,
                "schema": response_model.model_json_schema()
            }
        },
        "messages": [
            {
                "role": "system",
                "content": "You are administering a Big Five inventory. Return strict JSON only."
            },
            {
                "role": "user",
                "content": (
                    "Given the life story below, answer each inventory item as this person on a 1-5 Likert scale "
                    "(1=strongly disagree, 5=strongly agree).\n"
                    "Return a single JSON object mapping each item id to an integer 1..5. No extra text.\n\n"
                    f"LIFE STORY:\n{life_story}\n\n"
                    f"ITEMS:\n" + "\n".join(item_lines)
                )
            }
        ]
    }
    response = litellm.completion(**payload)
    text = response["choices"][0]["message"]["content"].strip()
    if text == "":
        raise RuntimeError("BFI response was empty. Expected strict JSON object mapping item ids to integers 1..5.")

    parsed = response_model.model_validate_json(text)

    answers = {}
    for item in items:
        item_id = str(item["id"])
        value = int(getattr(parsed, item_id))
        answers[item_id] = value

    return answers


def compute_traits_from_bfi(items: list[dict], responses: dict[str, int]) -> dict[str, float]:
    by_trait = {"O": [], "C": [], "E": [], "A": [], "N": []}

    for item in items:
        trait = str(item["factor"])
        item_id = str(item["id"])
        keyed = str(item["keyed"])
        raw = float(responses[item_id])
        scored = raw if keyed == "+" else (6.0 - raw)
        by_trait[trait].append(scored)

    traits = {}

    for trait, values in by_trait.items():
        mean_score = sum(values) / len(values)
        traits[trait] = float(mean_score - 3.0)

    return traits


def generate_bio(name: str, expanded_profile: str, life_story: str, traits: dict[str, float], model: str) -> str:
    trait_text = ", ".join(f"{trait}={value:.2f}" for trait, value in traits.items())
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Write concise third-person scientist profiles for behavioral simulation."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Profile name: {name}. Big Five (BFI-derived): {trait_text}.\n"
                    "Using the persona description and life story below, write a 120-160 word simulation profile focusing on communication and decision style.\n"
                    "Keep the same person identity and do not mention questionnaire items or numeric answers.\n\n"
                    f"PERSONA DESCRIPTION:\n{expanded_profile}\n\n"
                    f"LIFE STORY:\n{life_story}"
                )
            }
        ]
    }
    response = litellm.completion(**payload)
    bio = response["choices"][0]["message"]["content"].strip()
    return bio


async def build_population(cfg: DictConfig, base_dir: Path) -> list[dict[str, object]]:
    survey_path = base_dir / str(cfg.data.survey_path)
    survey = json.loads(survey_path.read_text())
    items = list(survey["items"])
    trait_names = list(survey["traits"])
    sampling = dict(survey["trait_sampling"])

    def build_one(index_and_traits: tuple[int, dict[str, float]]) -> dict[str, object]:
        index, sampled_traits = index_and_traits
        name = f"Persona-{index:03d}"

        profile_seed = build_seed_profile(index, int(cfg.population.seed))
        expanded_profile = expand_profile(profile_seed, str(cfg.population.bio_model), rounds=3)
        life_story = generate_life_story(name, expanded_profile, str(cfg.population.bio_model), target_traits=sampled_traits)
        responses = administer_bfi(life_story, items, str(cfg.population.bio_model))
        trait_values = compute_traits_from_bfi(items, responses)
        bio = generate_bio(name, expanded_profile, life_story, trait_values, str(cfg.population.bio_model))

        return {
            "name": name,
            "traits": trait_values,
            "bio": bio,
            "life_story": life_story,
            "survey_responses": responses
        }

    indexed_samples = []
    for index in range(1, int(cfg.population.n) + 1):
        sampled_traits = sample_traits(trait_names, sampling, int(cfg.population.seed) + index * 1000)
        indexed_samples.append((index, sampled_traits))

    personas = await to_thread_map(
        build_one,
        indexed_samples,
        max_concurrency=int(cfg.population.max_parallel_bio_workers)
    )
    return personas


if __name__ == "__main__":
    main()
