"""Exercise Omni's published context and model interfaces from sibling checkouts."""

import argparse
import json
import math
import os
import re
import struct
import subprocess
import sys
from hashlib import file_digest
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

REPOS = ("subnext", "unifur", "peregrine")
CITATION = re.compile(
    r"https://github\.com/s-b-repo/(subnext|unifur|peregrine)/blob/"
    r"([0-9a-f]{40})/([^\s)#]+)#L([1-9][0-9]*)"
)
WIDTH = 256
QUESTION = "Where does the North Library keep the blue atlas?"
SOURCE = "The North Library keeps the blue atlas in the map room."
THREADS = {"RAYON_NUM_THREADS": "1", "MATMUL_NUM_THREADS": "1"}


class Finding(ValueError):
    """A checked interface or citation violated its contract."""


def require(condition, message):
    if not condition:
        raise Finding(message)


def run(*argv, cwd=None, stdin=None, env=None):
    result = subprocess.run(
        argv,
        check=False,
        cwd=cwd,
        input=stdin,
        text=True,
        capture_output=True,
        env=dict(os.environ, **(env or {})),
        timeout=900,
    )
    if result.returncode:
        raise Finding(
            f"{Path(argv[0]).name} exited {result.returncode}\n{result.stderr or result.stdout}"
        )
    return result.stdout


def check_ledger(text, source):
    rows = [row for row in text.splitlines() if re.match(r"^\| N-\d+ \|", row)]
    require(len(rows) >= 3, "the ledger needs at least three experiment rows")
    seen = set()
    for row in rows:
        cells = [item.strip() for item in row.split("|")]
        require(
            len(cells) == 7 and not (cells[0] or cells[-1]), "invalid ledger row shape"
        )
        identifier, _, _, citations, reopening = cells[1:6]
        require(identifier not in seen, f"duplicate ledger ID {identifier}")
        seen.add(identifier)
        require(
            reopening.casefold() not in {"", "-", "--", "—", "tbd", "n/a"},
            f"{identifier} needs a reopening condition",
        )
        require(
            CITATION.search(citations), f"{identifier} needs a revision-pinned source"
        )
    references = set(CITATION.findall(text))
    files = {}
    for repo, revision, path, line in sorted(references):
        parts = PurePosixPath(path)
        require(
            not parts.is_absolute() and ".." not in parts.parts,
            f"invalid source path {path}",
        )
        key = repo, revision, path
        if key not in files:
            files[key] = source(*key).splitlines()
        number = int(line)
        lines = files[key]
        require(
            number <= len(lines) and lines[number - 1].strip(),
            f"{repo}@{revision[:7]}/{path}:{number} has no cited text",
        )
    return len(rows), len(references)


def binary(repo, name, package=None):
    override = os.environ.get(f"VCX_{name.upper().replace('-', '_')}")
    if override:
        executable = Path(override).resolve()
    else:
        target = repo / "target"
        command = ["cargo", "build", "--locked", "--bin", name, "--target-dir", target]
        if package:
            command += ["--package", package]
        run(*command, cwd=repo)
        executable = target / "debug" / name
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise FileNotFoundError(f"missing executable {executable}")
    return executable


def hash_file(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def make_corpus(directory):
    directory.mkdir()
    counts, hashes = {}, {}
    for name, prompts in (
        ("train", ["atlas", "globe", "catalog", "chart"]),
        ("validation", ["map", "journal"]),
    ):
        counts[name] = {"compatibility": len(prompts)}
        codec, aligned = bytearray(), bytearray()
        for item in prompts:
            prompt = [*f"Locate the {item}.".encode(), 257]
            reply = [*item.encode(), 257]
            target = struct.pack("<8H", *reply, *([258] * (8 - len(reply))))
            aligned.extend(
                struct.pack(f"<{WIDTH}H", *prompt, *([258] * (WIDTH - len(prompt))))
            )
            aligned.extend(target)
            codec.extend(target)
        for kind, content in (("aligned", aligned), ("codec", codec)):
            path = directory / f"{name}.{kind}.bin"
            path.write_bytes(content)
            hashes[path.name] = hash_file(path)
    manifest = {
        "prompt_width_bytes": WIDTH,
        "response_width_bytes_including_eos": 8,
        "packed_by_split_and_field": counts,
        "corpus_hashes": hashes,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def train_fixture(unifur, dblocks, root):
    corpus, output = root / "corpus", root / "weights"
    make_corpus(corpus)
    worker = unifur / "experiments/model_matrix/nina_worker.py"
    weights = []
    for stage in ("codec", "generator"):
        output_text = run(
            sys.executable,
            worker,
            "fit",
            "--binary",
            dblocks,
            "--input",
            corpus,
            "--out",
            output,
            "--profile",
            "preflight",
            "--stage",
            stage,
            "--steps",
            "2",
            "--batch-size",
            "2",
            "--checkpoint-every",
            "2",
            env=THREADS,
        )
        report = json.loads(output_text.strip().split("\n")[-1])
        require(report["step"] == 2, f"compatibility {stage} did not finish")
        weights.append(Path(report["checkpoint"]))
    return weights


def check_hits(hits):
    require(
        isinstance(hits, list) and 0 < len(hits) <= 8,
        "Subnext returned no bounded evidence",
    )
    for hit in hits:
        require(isinstance(hit, dict), "Subnext evidence must be a JSON object")
        require(
            type(hit.get("version")) is int and hit["version"] == 1,
            "Subnext evidence version must be 1",
        )
        require(
            all(isinstance(hit.get(key), str) and hit[key] for key in ("id", "text")),
            "Subnext evidence needs an ID and text",
        )
        require(
            type(hit.get("score")) in (int, float) and math.isfinite(hit["score"]),
            "Subnext evidence needs a finite score",
        )
        provenance = hit.get("provenance")
        require(
            isinstance(provenance, dict)
            and all(
                isinstance(provenance.get(key), str) and provenance[key]
                for key in ("source", "path")
            ),
            "Subnext evidence needs a provenance source and path",
        )
    require(
        len({hit["id"] for hit in hits}) == len(hits),
        "Subnext returned repeated evidence IDs",
    )


def check_generation(report, prompt_bytes):
    require(
        report["prompt_bytes"] == prompt_bytes
        and report["level_calls"] == 3
        and report["trunk_evaluations"] == 6
        and len(report["text"]) == 1
        and isinstance(report["text"][0], str),
        "Unifur returned an unexpected prompt width or latent generation report",
    )


def snapshot(store):
    return {
        path.relative_to(store): hash_file(path)
        for path in store.rglob("*")
        if path.is_file()
    }


def live(omni, repos, deployed_contract):
    with TemporaryDirectory(prefix="omni-cross-repo-") as temporary:
        root = Path(temporary)
        # Build from the checked-out sources unless the caller supplies binaries.
        verify = binary(omni, "omni-verify", "omni-verify")
        harness = binary(omni, "omni")
        dcr = binary(repos["subnext"], "dcr")
        dblocks = binary(repos["unifur"], "dblocks")
        run(verify, "verify", "--group", "contract")
        canonical = root / "contract.json"
        canonical.write_text(run(dblocks, "interop-contract"), encoding="utf-8")
        run(verify, "peer-contract", canonical)
        if deployed_contract:
            run(verify, "peer-contract", deployed_contract)
        document = json.loads(canonical.read_text(encoding="utf-8"))
        document["messages"][0]["fields"].append("unexpected_field")
        canonical.write_text(json.dumps(document), encoding="utf-8")
        try:
            run(verify, "peer-contract", canonical)
        except Finding as error:
            require(
                "fields differ" in str(error),
                f"peer manifest probe failed for the wrong reason: {error}",
            )
        else:
            raise Finding("Omni accepted a mismatched file-channel field")
        codec, generator = train_fixture(repos["unifur"], dblocks, root)
        source, store = root / "library.txt", root / "store"
        source.write_text(SOURCE + "\n", encoding="utf-8")
        run(dcr, "ingest", "--store", store, source)
        before = snapshot(store)
        require(before, "Subnext wrote no fixture store")
        query = f"Question:\n{QUESTION}\nAnswer:\n"
        hits = json.loads(
            run(
                dcr,
                "context-read",
                "--store",
                store,
                "--budget",
                "96",
                stdin=QUESTION,
                env={
                    "DCR_EVIDENCE_BYTES": str(
                        WIDTH - len(query.encode()) - len("Evidence:\n")
                    )
                },
            )
        )
        check_hits(hits)
        base = (
            harness,
            "model",
            "nina",
            "--binary",
            dblocks,
            "--backend",
            "cpu",
            "--tokenizer",
            codec,
            "--generator",
            generator,
        )
        control = json.loads(run(*base, stdin=QUESTION, env=THREADS))
        check_generation(control["generation"], len(query.encode()))
        require(
            control["context"]["used_ids"] == [],
            "control unexpectedly used external evidence",
        )
        result = json.loads(
            run(
                *base,
                "--dcr",
                dcr,
                "--context-store",
                store,
                "--context-budget",
                "96",
                stdin=QUESTION,
                env=THREADS,
            )
        )
        selected = result["context"]["used_ids"]
        by_id = {item["id"]: item["text"].strip() for item in hits}
        require(
            selected and all(item in by_id for item in selected),
            "Omni used no verified Subnext evidence",
        )
        require(
            any("blue atlas" in by_id[item].lower() for item in selected),
            "Omni omitted the matching source",
        )
        evidence = "Evidence:\n" + "".join(
            json.dumps(by_id[item], ensure_ascii=False, separators=(",", ":")) + "\n"
            for item in selected
        )
        prompt_bytes = len((evidence + query).encode())
        require(
            prompt_bytes <= WIDTH, "working set exceeds the checkpoint prefix width"
        )
        check_generation(result["generation"], prompt_bytes)
        require(
            snapshot(store) == before,
            "Omni's read-only route modified the Subnext store",
        )
        empty = json.loads(
            run(
                dcr,
                "context-read",
                "--store",
                store,
                "--budget",
                "96",
                stdin=QUESTION,
                env={"DCR_EVIDENCE_BYTES": "0"},
            )
        )
        require(empty == [], "Subnext returned evidence under a zero-byte allowance")
        require(
            snapshot(store) == before, "Subnext's zero-budget read modified the store"
        )


def self_test():
    citation = "https://github.com/s-b-repo/subnext/blob/" + "1" * 40 + "/TODO.md#L2"
    ledger = "\n".join(
        f"| N-0{i} | claim | result | {citation} | repeat on a larger input |"
        for i in range(1, 4)
    )
    source = lambda *_: "heading\nevidence\n"
    require(check_ledger(ledger, source) == (3, 1), "positive ledger control")
    hit = {
        "version": 1,
        "id": "north",
        "text": SOURCE,
        "score": 0.5,
        "provenance": {"source": "subnext", "path": "library"},
    }
    check_hits([hit])
    checks = (
        (
            "duplicate ledger ID",
            lambda: check_ledger(ledger.replace("N-03", "N-02"), source),
        ),
        ("no cited text", lambda: check_ledger(ledger.replace("#L2", "#L3"), source)),
        (
            "reopening condition",
            lambda: check_ledger(
                ledger.replace("repeat on a larger input", "TBD", 1), source
            ),
        ),
        (
            "revision-pinned",
            lambda: check_ledger(ledger.replace(citation, "old path"), source),
        ),
        ("version", lambda: check_hits([dict(hit, version=2)])),
        (
            "version",
            lambda: check_hits(
                [{key: val for key, val in hit.items() if key != "version"}]
            ),
        ),
        ("provenance", lambda: check_hits([dict(hit, provenance={})])),
    )
    for message, action in checks:
        try:
            action()
        except Finding as failure:
            require(
                message in str(failure),
                f"self-test failed for the wrong reason: {failure}",
            )
        else:
            raise Finding(f"self-test accepted a broken {message}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--ledger-only", action="store_true")
    parser.add_argument("--deployed-contract", type=Path)
    args = parser.parse_args()
    try:
        if args.self_test:
            self_test()
            if not args.quiet:
                print("cross-repo self-test passed")
            return 0
        omni = Path(
            os.environ.get("VCX_ROOT", Path(__file__).resolve().parents[1])
        ).resolve()
        repos = {
            name: Path(
                os.environ.get(f"VCX_{name.upper()}", omni.parent / name)
            ).resolve()
            for name in REPOS
        }
        for path in repos.values():
            if not (path / "Cargo.toml").is_file():
                raise FileNotFoundError(f"expected sibling checkout at {path}")
        ledger = Path(os.environ.get("VCX_LEDGER", omni / "docs/NEGATIVE-RESULTS.md"))
        rows, citations = check_ledger(
            ledger.read_text(encoding="utf-8"),
            lambda repo, revision, path: run(
                "git", "show", f"{revision}:{path}", cwd=repos[repo]
            ),
        )
        if not args.ledger_only:
            live(omni, repos, args.deployed_contract)
        if not args.quiet:
            print(
                f"cross-repo gate passed with {rows} experiment rows and {citations} pinned citations"
            )
    except (OSError, subprocess.TimeoutExpired) as failure:
        print(f"verify-cross-repo: could not run: {failure}", file=sys.stderr)
        return 3
    except (ValueError, KeyError, TypeError) as failure:
        print(f"verify-cross-repo: FAIL: {failure}", file=sys.stderr)
        return int(args.strict or args.self_test)
    return 0


if __name__ == "__main__":
    sys.exit(main())
