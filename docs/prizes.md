# AI Mathematical Olympiad - Progress Prize 3 - Prizes

## TOTAL FUND FOR PROGRESS PRIZE 3: $2,207,152

Awards Points & Medals

## Prizes for Top-Ranking Teams in this Competition:

- 1st Place: $262,144
- 2nd Place: $131,072
- 3rd Place: $65,536
- 4th Place: $32,768
- 5th Place: $16,384

## Overall Progress Prize Winner

The Overall Progress Prize Winner shall be the highest ranking team that achieves a score of at least 47/50 on both public and private test sets. After any prizes for the five top-ranking teams have been awarded, the remainder of the total fund shall be awarded to the Overall Progress Prize Winner.

If a team is named the Overall Progress Prize Winner in this competition, the prize will be at least $1,589,248. If no team is named the Overall Progress Prize Winner in this competition, the remainder of the total fund shall roll over to the next competition, where the same prize allocation will apply.

## Additional Prizes: $110,000

- **Longest Leader Prize**: $20,000
- **Hard Problem Prize**: $30,000
- **Math Corpus Prize**: $30,000
- **Writeup Prizes**: 2x $15,000

### Longest Leader Prize: $20,000

Awarded to the team whose notebook(s) generates the best scoring submission on the leaderboard for the longest period of time between November 20, 2025 and February 2, 2026 11:59 PM UTC. The notebooks need to adhere to the same requirements and restrictions regarding licensing, reproducibility, and documentation to which the winning Submission is subject (see Competition Rules). The notebook(s) and any datasets used must be made publicly available at latest at February 9, 2026 11:59 PM UTC and kept public until the final Progress Prizes are awarded to the winning Teams at the end of the competition. In the event that a team has multiple public notebooks, each achieving the top leaderboard position at different times, those durations will be combined.

### Hard Problem Prize: $30,000

Awarded to the highest ranked team(s) on the private leaderboard who solved the most difficult problem(s) in the private test set. Problem difficulty is measured by the average accuracy score across all selected submissions at the end of the competition, where average accuracy is calculated as described in the Evaluation section (accounting for both runs of each submission). For example, if the most difficult problem has an average accuracy of 1.7%, then the highest ranked team on the private leaderboard who achieved a full score on that problem will receive the prize. If multiple problems tie for "most difficult": We will identify the highest ranked team for each tied problem separately, and the $30,000 prize will be split equally among the winning teams. For example, if two problems tie for most difficult, each with its own highest-scoring solver, the prize will be divided into two $15,000 awards. In the event of a tie among notebooks, the tiebreaker will be based on submission time (first to submit).

### Math Corpus Prize: $30,000

Awarded to the team who provides the most valuable dataset to the community for improving AIMO model performance.

**Math Corpus Prize Requirements:**

- The dataset must be publicly released either on Kaggle or on HuggingFace prior to February 9, 2026 11:59 PM UTC. Teams must create a Kaggle Discussion post until this date that explicitly links to their dataset and tags it as an entry for the Math Corpus Prize. All eligible datasets will be judged according to the criteria below, and the highest scoring dataset will be awarded the prize.
- Your Kaggle Discussion post will be used to judge your dataset submission based on the criteria below, and your claims will be independently verified by the Host Team.
- The dataset must be in English.
- The earliest timestamped version of the dataset will be used for evaluation (later modifications will not be considered)
- The dataset is not allowed to exceed 5M datapoints, which is an upper bound on common datasets used by prior AIMO winners to fine-tune their models. Each datapoint is not allowed to exceed more than 100k characters.
- Each datapoint from the submitted dataset must come with an open source license that allows free dissemination of data.

**Math Corpus Prize Evaluation Criteria:**

- Data Novelty (25 points)
  - Is the dataset distinctly unique from other datasets found on the internet? Minor modification are not considered novel. In particular, translations into English from other languages are not considered significant modifications.
  - Describe what methods you used to ensure your dataset is novel, and note how your dataset does not overlap with prior submitted datasets.
- Format (25 points)
  - Does the dataset comes in a format that makes it easy to handle and train a model?
  - Does each datapoint contain rich metadata?
- Performance (50 points)
  - Does the dataset improve mathematical reasoning and aid in improving model performance during the AIMO3. How so and how much?

### Writeup Prize: $30,000 (2 awards x $15,000)

Awarded to the two best Solution Writeups from non-winning teams based on the below criteria.

**Writeup Prize Requirements:**

- The Writeup should be an official Kaggle Writeup (https://www.kaggle.com/discussions/product-announcements/593763), attached to your submission within one week after the competition ended.
- It should mirror an academic publication about your submission, and include graphs that outline how your model was trained, the impact of different techniques and methods on performance, and how it could be further improved. (The arXiv post by the AIMO2 winners is a good reference paper in terms of academic standard: https://arxiv.org/abs/2504.16891)
- It should be limited to 30k words (which approximately corresponds to a 50-page paper), and split into a main section of not more than 5k words (approximately 10 pages), which outlines your methodology, and the remaining sections, which include the bibliography, examples, and any other materials that support the main section.

**Writeup Prize Evaluation Criteria:**

- Clarity (10 points)
  - Does the Writeup describe in detail the entire model creation lifecycle (including data collection, training, and benchmarking), as well as any model variants you explored?
- Ablation Studies and Variants (10 points)
  - Does the Writeup contain information that pinpoints where the performance gains of your model came from?
- Comparison with the State of the Art (10 points)
  - Does the Writeup evaluate how your model (potentially split across types of reasoning modes or mathematical problems) compares to the state of the art for both open-weight and commercial models?
- Graphs and Charts (10 points)
  - Does the Writeup contain well-designed visuals that clearly "tell the story" by illustrating your model's performance, training process, and key experimental results?
- Reproducibility (60 points)
  - Is your write-up sufficiently detailed to easily allow the reader to reproduce the training of your model starting from the same artifacts (base open-weight models and data) that you started from?
