# Addendum 2: scope of the full runs after screening

Frozen 2026-09-29, while the screening diagnostic pods are still running, so before any screening diagnostic value and before any SDrop run on the screened datasets.

- **Full runs:** in9bg, counteranimal and fairface_age get full runs whatever their predictions:
  - 6 arms: ERM, Energy, Random, Shuffle, Energy top-k, Shuffle top-k;
  - 6 seeds, as in the addendum, section B.
- **celeba_wavy and celeba_glasses:** screened with the diagnostic, and their predictions are recorded. They are run in full **only if the budget allows after the three datasets above**. They are not independent of the CelebA benchmark already in the paper (same images), and one full run of each costs about three times one of the others.
  - If they are not run in full, the paper says so and gives only their diagnostic values.
  - The decision to run them does not depend on their diagnostic values or on any result. It depends only on the remaining budget once the three datasets are finished.
- **Everything else** in the addendum is unchanged: the prediction rule, all outcomes reported, and BH within the confirmatory family.
