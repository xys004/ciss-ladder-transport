# Publication Checklist

Use this list before making the repository public and linking it to Zenodo.

## Metadata

- Confirm the final repository title.
- Confirm author order in `CITATION.cff`.
- Add ORCIDs if the authors want them in the public record.
- Add the final article DOI once available.

## Legal

- Confirm that all authors agree with the chosen MIT License and repository contents.

## Scientific reproducibility

- Decide whether the public release should archive only code or also selected raw/processed datasets.
- Confirm the intended chemical-potential convention in the current-integration step.
- Confirm whether the hardcoded angular increment in the legacy transport scripts is part of the model definition.

## Repository hygiene

- Review the `legacy/` copies one last time to ensure nothing relevant was omitted.
- Add one example dataset if you want users to run the post-processing immediately after cloning.
- Create a tagged release, then connect that release to Zenodo.
- Review `.zenodo.json` and update affiliations, ORCIDs, and final wording if needed.
