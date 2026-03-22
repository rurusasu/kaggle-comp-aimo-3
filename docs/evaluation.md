# AI Mathematical Olympiad - Progress Prize 3 - Evaluation

## Evaluation Metric

During the submission period, submission notebooks are run only over the public test set and are evaluated by the unnormalized **accuracy** (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html) between their predicted labels and the ground-truth labels (i.e. number of correct answers).

After the submission deadline, submission notebooks will be run *twice* over the private test set and their predictions concatenated into a single submission file. We then evaluate submissions by a **penalized accuracy** score, as follows:

- If both predicted answers for a problem are correct, the score is 1 for that problem.
- If one predicted answer is correct and the other is incorrect, the score is 0.5 for that problem.
- If neither predicted answer is correct, the score is 0 for that problem.

A submission's overall score is the sum of its scores for each problem.

## Answer Format

In this competition, every ground-truth label is an integer between 0 and 99999, inclusive. Any modulo calculation required is explicitly stated in the problem statement (e.g., "What is the remainder when N is divided by 10^5?"), meaning all problems have answers in this range without any further adjustments.

**Note:** This is a change from the first and second Progress Prizes, which used 3-digit answers with an implicit requirement to take your answer modulo 1000.

Answers may require basic computations, including modular arithmetic. For example, floor(10^4 * sqrt(2)) = 14142 and 3^2025 mod 10^5 = 29443. Additional examples of computations that may be required are provided in the Reference Problems PDF found on the Data page.

## Submitting

You must submit to this competition using the provided Python evaluation API, which serves test set instances one-by-one in random order for the public leaderboard, and uses fixed, random order for the private leaderboard. To use the API, follow the template in this notebook: https://www.kaggle.com/code/ryanholbrook/aimo-3-submission-demo
