"""Three-way citation accuracy with an open-weight LLM, against the paper's GPT-4 test.

Two prompts, fixed before any test citation is judged:
  paper     verbatim from the paper's Supplementary Material, section F (task text and
            its four train demonstrations), so differences from GPT-4 come from the
            model and not from the instructions
  improved  addresses what Table S2 shows went wrong (GPT-4: NOT_ACCURATE recall 0.05,
            IRRELEVANT recall 0.74, precision 0.35). The paper's prompt calls anything
            "not explicitly discussed in the evidence" NOT ENOUGH INFORMATION, which
            also describes NOT_SUBSTANTIATE, and its demonstrations show only CONTRADICT
            and MISQUOTE errors. This one separates the classes using the annotation
            guidelines (sections C and D) and demonstrates every error type with the
            supplement's own examples, all train citations. It also returns scores.

Choose the prompt on dev; test is judged once per prompt and evidence pair:
  python llm_threeclass.py --dry-run --prompt improved --evidence topk/top20 --split dev
  python llm_threeclass.py --prompt improved --evidence topk/top20 --split dev    # needs GPUs
  python llm_threeclass.py --model Qwen/Qwen3.8-27B --prompt improved --evidence topk/top20 --split dev
  python llm_threeclass.py --analyze
Other models reuse the prompt chosen with gpt-oss-20b on dev.
"""
import argparse
import glob
import json
import re
import time
from pathlib import Path

from llm_oversimplify import ROOT, read_jsonl

# Per model: output folder, response format, chat-template options and decoding. gpt-oss keeps
# the folder and settings of its runs. Qwen recommends sampling for thinking mode, so its draws
# are seeded per citation; the others decode greedily.
MODELS = {
    "openai/gpt-oss-20b": {"slug": "gptoss20b", "format": "harmony", "loader": "mxfp4", "max_new_tokens": 4096,
                           "template": lambda effort: {"reasoning_effort": effort}, "sampling": None},
    "Qwen/Qwen3.8-27B": {"slug": "qwen38-27b", "format": "think", "loader": "image-text", "max_new_tokens": 8192,
                         "template": lambda effort: {"reasoning_effort": effort},
                         "sampling": {"do_sample": True, "temperature": 1.0, "top_p": 0.95, "top_k": 20}},
    "google/medgemma-27b-text-it": {"slug": "medgemma27b", "format": "plain", "loader": "causal", "max_new_tokens": 512,
                                    "template": lambda effort: {}, "sampling": None},
}
LABELS = ["ACCURATE", "NOT_ACCURATE", "IRRELEVANT"]
FINAL = re.compile(r"<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>|$)", re.S)

PAPER_TASK = """You are assessing citations in biomedical articles for accuracy. That means checking whether the citation in a citing article accurately reflects the contents of the reference article.

An ACCURATE citation is one in which the information provided aligns well with facts, evidence, and established knowledge. The exact details should match explicitly and not just the general theme for it to be ACCURATE.
A NOT_ACCURATE citation is one in which the citation context contradicts the passage from the reference article, or does not substantiate it, or oversimplifies it, or distorts its meaning, or misquotes it.
A NOT ENOUGH INFORMATION citation is one where the reference article is irrelevant to the citation i.e. the citation content refers to some information that is not explicitly discussed in the evidence.

I will provide some text from the citing article within the <context></context> tags and highlight the citation marker within that context as <|cit|> (single citation) or <|multi_cit|> (multi-citation). There might be other citations in the same context which I will highlight as <|other_cit|> that are not relevant to your assessment. Then I will provide a few sentences from the reference article as evidence within <evidence></evidence> tags. Each sentence in evidence will be marked indexed by a number in parentheses. The prediction as ACCURATE, NOT_ACCURATE, or NOT ENOUGH INFORMATION will follow in the <prediction></prediction> tags as in the examples below.

# Example
<context>
Disease related studies that involve microbiome analysis using metagenomics include obesity [<|other_cit|>], Crohn's disease [<|multi_cit|>], type 2 diabetes [<|other_cit|>] and many others [<|other_cit|>].
</context>

<evidence>
[0] This study's inferred IBD metagenomes include broadly increased oxidative metabolism, decreased SCFA production, and increased mucin degradation relative to healthy subjects. [1] The microbiome of ileal Crohn's disease was notable for increases in virulence and secretion pathways. [2] An important consideration that informed the remainder of our analysis, and which is often overlooked in studies of the microbiome, was the consistent covariation among disease status, aspects of subject environment, and microbiome structure. [3] We have thus analyzed the GI microbiomes of 121 CD patients, 75 UC patients, and 27 healthy controls using a novel multivariate metagenomic analysis pipeline specifically accounting for environmental factors (including treatment, age, and tobacco use). [4] In addition to assessing microbiome composition, we have analyzed the inferred metagenome as determined from phylogenetically-associated reference genomes, including metabolic modules and pathways also associated with disease status and with environmental factors such as medications and smoking.
</evidence>

<prediction>
ACCURATE
</prediction>

<reason>
The context states that studies involving microbiome analysis using metagenomics include Crohn's disease. The evidence sentences support this by mentioning:
Analyzing the microbiome of Crohn's disease patients (sentences 1, 3)
Finding changes in metabolic pathways and microbiome composition associated with Crohn's disease status (sentences 2, 4)
Therefore, the citation accurately reflects that metagenomic studies have analyzed the microbiome in Crohn's disease. The evidence aligns with and substantiates the information provided in the citing context.
Since the claiming text's statement about Crohn's disease metagenomic studies is corroborated by the details in the evidence, I assess this citation as accurate. The citation context and evidence are in agreement.
</reason>

# Example
<context>
This is coherent with the fact that hACE2 expression were not observed in the gut of the mice used in that study [<|cit|>].
</context>

<evidence>
[0] Although hACE2 is expressed in the gastrointestinal tract in other hACE2-expressing mice, productive infection was observed only upon intragastric inoculation or at early time points following intranasal infection11,12. [1] In the gastrointestinal tract of K18-hACE2 mice, hACE2 was expressed most abundantly in the colon, which correlated with infection seen at later time points. [3] The spleen in SARS- CoV-2-infected K18-hACE2 mice appeared normal (Extended Data Fig 2f), and fibrin thrombi were not detected in any of the extra-pulmonary organs examined. [4] We saw substantial immune cell accumulation in the lungs of K18-hACE2 mice, an observation consistent with post-mortem analysis of human patients39. [5] mRNA expression levels of hACE2 in the lung, kidney, heart, brain, spleen, duodenum, colon, and ileum of naive K18 hACE2 mice (two experiments, n = 5).
</evidence>

<prediction>
NOT_ACCURATE
</prediction>

<reason>
The context states that hACE2 expression were not observed in the gut. But sentence [1] in the evidence states that hACE2 was expressed most abundantly in the colon. Since the colon is a part of the gut, the citation context contradicts the evidence and hence the citation is NOT_ACCURATE.
</reason>

# Example
<context>
All analyses were performed using R software version 3.0.2 (library survival) and Stata 13.0 (College Station, TX) [<|cit|>].
</context>

<evidence>
[0] ### Statistical Analyses [1] Sensitivity analyses: Restricting to cohorts with death from registries. [2] To further investigate this issue, we conducted sensitivity analyses taking into account loss to follow-up, and were reassured to find little differences in our estimates. [3] We calculated all-cause mortality rates (per 1,000 person-years) by dividing the total number of deaths by the total number of person-years of observation. [4] In effort to reduce bias from informative censoring, we used weighted regression methods to estimate mortality rates, using an indicator variable that was created to identify participants who were lost to follow-up.
</evidence>

<prediction>
NOT ENOUGH INFORMATION
</prediction>

<reason>
R or Stata was not mentioned in the evidence segment and there is no mention of the software used in the evidence segment. The evidence is not sufficient to decide if R and Strata were used and hence the citation is NOT ENOUGH INFORMATION.
</reason>

# Example
<context>
We have previously shown an even higher mortality rate, around 30 deaths per 100 person-years, during the pre-treatment screening period [<|other_cit|>] in this population and recent reviews show 32% loss to follow-up during this pre-treatment period despite eligibility for ART [<|cit|>], a 20% loss to follow-up among patients within the first year of antiretroviral therapy [<|other_cit|>], among whom mortality is very high [<|other_cit|>].
</context>

<evidence>
[0] While pre-ART loss to care may not pose as immediate a mortality threat as loss of patients who already have clinical AIDS, it is still a major impediment to improving the outcomes of HIV care and treatment overall, is itself a contributor to the high mortality observed during the first year on ART, and wastes scarce health system resources. [1] Studies of retention in pre-ART care report substantial loss of patients at every step, starting with patients who do not return for their initial CD4 count results and ending with those who do not initiate ART despite eligibility. [2] Difficult as this problem is for managing ART patients, it is even worse during the pre-ART period, because patients are expected to visit the clinic less frequently, and more clinics are able to provide pre-ART services than are accredited to offer ART. [3] In particular, the estimate of the overall loss of patients during pre-ART care is likely to be imprecise. [4] The rate of early mortality and loss to follow-up, which itself portends mortality for many, averages 23% across the region.
</evidence>

<prediction>
NOT_ACCURATE
</prediction>

<reason>
Sentence [4] states an average loss of 23% across the region which is misquoted as 32% in the citation context. Since, the numbers or percentages do not match, the citation is NOT_ACCURATE.
</reason>

"""

IMPROVED_INSTRUCTIONS = """You check whether a citation in a biomedical article faithfully reflects the paper it cites. You receive the citing text and excerpts retrieved from the cited paper. The excerpts are not the whole paper and may miss the relevant passage. The marker <|cit|> shows where the cited paper is referenced, <|multi_cit|> means it is cited together with other papers at that point, and <|other_cit|> refers to a different paper.

Assign one label, using the first category that applies:
- IRRELEVANT: the cited paper is about a different matter altogether, so the citation has nothing to do with its content.
- NOT_ACCURATE: the paper is relevant to the citation, but the citation has one of these errors:
  - CONTRADICT: it contradicts a statement in the paper.
  - NOT_SUBSTANTIATE: the paper fails to substantiate all statements made in the citing text. A citation with several claims is not substantiated if any of them is unsupported.
  - MISQUOTE: numbers or percentages are misquoted.
  - OVERSIMPLIFY: findings are oversimplified or overgeneralized, for example by dropping qualifications such as population, region or conditions.
  - ETIQUETTE: the citation is ambiguous and it is unclear what is being cited from this paper. This often occurs when it is one of several citations placed together after a statement covering more than this paper discusses.
- ACCURATE: the citing text is consistent with the paper. This includes paper statements that themselves report earlier work.

Do not give the citing authors the benefit of the doubt, and limit inference. Check each claim in the citing text separately against the excerpts. When a topic is discussed in the excerpts but a claim about it is not supported, that is NOT_SUBSTANTIATE rather than IRRELEVANT.

Reply with only a JSON object with exactly these keys: "explanation" (one sentence), "error_type" (one of "NONE", "CONTRADICT", "NOT_SUBSTANTIATE", "MISQUOTE", "OVERSIMPLIFY", "ETIQUETTE", "IRRELEVANT"), "accurate_score", "not_accurate_score", "irrelevant_score" (integers from 0 to 100 showing how strongly each label applies), and "label" ("ACCURATE", "NOT_ACCURATE" or "IRRELEVANT").

Examples, from the annotation guidelines:

Citing text: Patients receiving remdesivir had a shorter time to recovery (median time to recovery, 10 [95% CI, 9-11] vs 15 [95% CI, 13-18] days; rate ratio for recovery, 1.29 [95% CI, 1.12-1.49]) [<|cit|>].
Excerpt: Those who received remdesivir had a median recovery time of 10 days (95% confidence interval [CI], 9 to 11), as compared with 15 days (95% CI, 13 to 18) among those who received placebo (rate ratio for recovery, 1.29; 95% CI, 1.12 to 1.49).
Label: ACCURATE. The figures match.

Citing text: It is known that glycosylation has crucial roles in viral pathobiology, such as protein proper folding and viral tropism [<|cit|>].
Excerpt: Viral glycosylation has wide-ranging roles in viral pathobiology, including mediating protein folding and stability and shaping viral tropism (9).
Label: ACCURATE. The paper states this, although it reports earlier work.

Citing text: This is coherent with the fact that hACE2 expression were not observed in the gut of the mice used in that study [<|cit|>].
Excerpt: In the gastrointestinal tract of K18-hACE2 mice, hACE2 was expressed most abundantly in the colon, which correlated with infection seen at later time points.
Label: NOT_ACCURATE, CONTRADICT. The colon is part of the gut.

Citing text: Most studies have focused on the prevalence of high BMI, associating it with increased risk of conditions such as coronary heart disease, stroke, diabetes, and cancer [<|cit|>].
Excerpt: Metabolic mediators of the effects of body-mass index, overweight, and obesity on coronary heart disease and stroke: a pooled analysis of 97 prospective cohorts with 1.8 million participants.
Label: NOT_ACCURATE, NOT_SUBSTANTIATE. The paper does not examine the BMI-cancer association.

Citing text: Recent reviews show 32% loss to follow-up during this pre-treatment period despite eligibility for ART [<|cit|>].
Excerpt: The rate of early mortality and loss to follow-up, which itself portends mortality for many, averages 23% across the region.
Label: NOT_ACCURATE, MISQUOTE. 23% is misquoted as 32%.

Citing text: Combination antiretroviral therapy has changed HIV-1 infection from a life-threatening disease to a manageable chronic condition, and life expectancy of people living with HIV is approaching that of non-infected individuals [<|cit|>].
Excerpt: A 20-year-old HIV-positive adult on ART in the U.S. or Canada is expected to live into their early 70s, a life expectancy approaching that of the general population.
Label: NOT_ACCURATE, OVERSIMPLIFY. Age and region qualifications are discarded.

Citing text: The iSPNs are enriched in enkephalin and D2 type dopamine receptors, project exclusively to the GPe, and play a role in suppression of unwanted movements [<|multi_cit|>].
Excerpt: Only the role of iSPNs in suppressing unwanted movements is discussed.
Label: NOT_ACCURATE, ETIQUETTE. It is unclear which statements this paper is cited for.

Citing text: All analyses were performed using R software version 3.0.2 (library survival) and Stata 13.0 (College Station, TX) [<|cit|>].
Excerpt: Closing the Gap: Increases in Life Expectancy among Treated HIV-Positive Individuals in the United States and Canada.
Label: IRRELEVANT. The paper has nothing to do with this software; the authors likely cited the wrong reference."""


# Chosen on dev (improved prompt, MonoT5 top-20) before any test run of Qwen or MedGemma: Qwen almost never
# answers IRRELEVANT but gives gold IRRELEVANT citations irrelevant_score 10-15; 11-15 is best for gpt-oss and Qwen.
IRR_THRESHOLD = 15


def gold_label(claim):
    labels = {e["label"] for es in claim["evidence"].values() for e in es}
    return labels.pop() if labels else "IRRELEVANT"


def build(prompt, evidence, split):
    "Requests in file order. The gold label is kept aside, never in the messages."
    docs = {d["doc_id"]: d["abstract"] for d in read_jsonl(ROOT / "data" / evidence / "corpus.jsonl")}
    requests = []
    for c in read_jsonl(ROOT / "data" / evidence / f"claims_{split}.jsonl"):
        sentences = " ".join(f"[{i}] {s}" for i, s in enumerate(docs[c["cited_doc_ids"][0]]))
        case = f"# Test Case\n<context>\n{c['claim']}\n</context>\n\n<evidence>\n{sentences}\n</evidence>"
        if prompt == "paper":
            messages = [{"role": "user", "content": PAPER_TASK + case}]
        else:
            messages = [{"role": "developer", "content": IMPROVED_INSTRUCTIONS}, {"role": "user", "content": case}]
        requests.append({"id": c["id"], "gold": gold_label(c), "messages": messages})
    return requests


def out_dir(model):
    return ROOT / "runs" / f"llm-threeclass-{MODELS[model]['slug']}"


def final_text(fmt, text):
    "The answer part of a response, or None when the model never finished thinking."
    if fmt == "harmony":
        final = FINAL.search(text)
        return final.group(1) if final else None
    if fmt == "think":
        return text.rsplit("</think>", 1)[1] if "</think>" in text else None
    return text


def parse_answer(prompt, body):
    "The judgment from the answer text, or None if it is missing or malformed."
    if body is None:
        return None
    if prompt == "paper":
        found = re.search(r"<prediction>\s*(ACCURATE|NOT_ACCURATE|NOT ENOUGH INFORMATION)\s*<", body)
        if not found:
            return None
        return {"label": "IRRELEVANT" if found.group(1) == "NOT ENOUGH INFORMATION" else found.group(1),
                "explanation": body[found.end():].strip()[:500]}
    try:
        answer = json.loads(body[body.find("{"): body.rfind("}") + 1])
        scores = [int(answer[k]) for k in ("accurate_score", "not_accurate_score", "irrelevant_score")]
    except (ValueError, KeyError, TypeError):
        return None
    if answer.get("label") not in LABELS or not all(0 <= s <= 100 for s in scores):
        return None
    return {"label": answer["label"], "error_type": answer.get("error_type"), "explanation": str(answer.get("explanation", "")),
            "scores": dict(zip(LABELS, scores))}


def load(model_name):
    import torch

    spec = MODELS[model_name]
    if spec["loader"] == "mxfp4":
        from llm_local import load_model
        return load_model(model_name)
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

    cls = AutoModelForImageTextToText if spec["loader"] == "image-text" else AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(model_name)
    model = cls.from_pretrained(model_name, dtype=torch.bfloat16, device_map="auto",
                                max_memory={0: "40GiB", 1: "40GiB"}).eval()
    print("loaded, GiB per GPU:", [round(torch.cuda.memory_allocated(i) / 2**30, 1) for i in (0, 1)], flush=True)
    return model, tok


def judge(model, tok, req, prompt, model_name, effort, max_new_tokens, seed):
    import torch

    spec = MODELS[model_name]
    messages = req["messages"]
    if spec["format"] != "harmony":  # only gpt-oss has a developer role
        messages = [dict(m, role="system") if m["role"] == "developer" else m for m in messages]
    ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True,
                                  **spec["template"](effort)).to(model.device)
    decoding = spec["sampling"] or {"do_sample": False}
    if spec["sampling"]:
        torch.manual_seed(seed * 1_000_003 + req["id"])
    with torch.inference_mode():
        out = model.generate(**ids, max_new_tokens=max_new_tokens, **decoding)
    new = out[0, ids["input_ids"].shape[1]:]
    text = tok.decode(new, skip_special_tokens=spec["format"] != "harmony")
    answer = parse_answer(prompt, final_text(spec["format"], text))
    stop = "end_turn" if answer else ("max_tokens" if len(new) >= max_new_tokens else "parse_error")
    row = {"id": req["id"], "gold": req["gold"], "stop_reason": stop,
           "input_tokens": int(ids["input_ids"].shape[1]), "output_tokens": len(new)}
    row.update(answer or {"raw": text[-2000:]})
    return row


def run(args):
    out = out_dir(args.model)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.split}-{args.prompt}-{args.evidence.replace('/', '_')}.jsonl"
    done = {r["id"] for r in read_jsonl(path)} if path.exists() else set()
    todo = [r for r in build(args.prompt, args.evidence, args.split) if r["id"] not in done][: args.limit]
    print(f"{args.model} {path.name}: {len(done)} already done, {len(todo)} to run", flush=True)
    model, tok = load(args.model)
    max_new_tokens = args.max_new_tokens or MODELS[args.model]["max_new_tokens"]
    t0 = time.time()
    with open(path, "a") as f:
        for n, req in enumerate(todo, 1):
            row = judge(model, tok, req, args.prompt, args.model, args.effort, max_new_tokens, args.seed)
            f.write(json.dumps(row) + "\n")
            f.flush()
            if n % 20 == 0 or n <= 3:
                print(f"{n}/{len(todo)} {(time.time() - t0) / n:.1f} s each, last: {row['stop_reason']}, "
                      f"{row['output_tokens']} tokens", flush=True)
    analyze()


def threshold_label(row):
    "The model's label, overridden by IRRELEVANT when irrelevant_score reaches IRR_THRESHOLD."
    if "scores" in row and row["scores"]["IRRELEVANT"] >= IRR_THRESHOLD:
        return "IRRELEVANT"
    return row.get("label", "UNJUDGED")


def analyze():
    from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score

    for path in sorted(glob.glob(str(ROOT / "runs/llm-threeclass-*/*.jsonl"))):
        rows = read_jsonl(path)
        ok = [r for r in rows if "label" in r]
        if not ok:
            continue
        name = f"{Path(path).parent.name[len('llm-threeclass-'):]}/{Path(path).stem}"
        gold = [r["gold"] for r in rows]
        # A citation without a parsed answer counts as wrong, not as absent.
        rules = {"label": [r.get("label", "UNJUDGED") for r in rows]}
        if any("scores" in r for r in rows):
            rules[f"irr>={IRR_THRESHOLD}"] = [threshold_label(r) for r in rows]
        for rule, pred in rules.items():
            p, r_, f, _ = precision_recall_fscore_support(gold, pred, labels=LABELS, zero_division=0)
            cells = " ".join(f"{k[:5]} {p[i]:.2f}/{r_[i]:.2f}/{f[i]:.2f}" for i, k in enumerate(LABELS))
            accuracy = sum(g == q for g, q in zip(gold, pred)) / len(rows)
            line = (f"{name} [{rule}]: {len(ok)}/{len(rows)} judged | micro {accuracy:.3f} "
                    f"macro {f1_score(gold, pred, labels=LABELS, average='macro'):.3f} | P/R/F1 {cells}")
            scored = [r for r in ok if r["gold"] in LABELS[:2] and "scores" in r]
            if scored and rule == "label":
                margin = [r["scores"]["NOT_ACCURATE"] - r["scores"]["ACCURATE"] for r in scored]
                line += f" | AUC {roc_auc_score([r['gold'] == 'NOT_ACCURATE' for r in scored], margin):.3f}"
            print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", choices=["paper", "improved"])
    ap.add_argument("--evidence", help="dataset under data/, e.g. topk/top20")
    ap.add_argument("--split", choices=["dev", "test"])
    ap.add_argument("--model", default="openai/gpt-oss-20b", choices=list(MODELS))
    ap.add_argument("--effort", default="medium", choices=["low", "medium", "high"])
    ap.add_argument("--max-new-tokens", type=int, default=None, help="default: per model, see MODELS")
    ap.add_argument("--seed", type=int, default=76, help="for models that sample")
    ap.add_argument("--limit", type=int, default=None, help="only the next N citations (speed checks)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    args = ap.parse_args()
    if args.analyze:
        analyze()
    elif args.dry_run:
        requests = build(args.prompt, args.evidence, args.split)
        chars = sum(len(m["content"]) for r in requests for m in r["messages"])
        print(f"{len(requests)} requests, about {chars / 4 / len(requests):,.0f} input tokens each")
        print(requests[0]["messages"][-1]["content"][-900:])
    else:
        run(args)


if __name__ == "__main__":
    main()
