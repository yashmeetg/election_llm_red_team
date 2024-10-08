# RedElect2024
Project completed during the AI Alignment course, aiming to test LLMs against prompts that elicit politically charged text related to the US election

## Intro + Research Question

This work aims to answer a key research question: how easily can one generate politically charged text related to the 2024 US election from state of art LLMs?

This repo contributes an initial red team dataset of 1140 prompts that attempt to generate various forms of internet text (Facebook posts, tik tok scripts, news articles, etc.) about the hottest topics surrounding the 
2024 US election. This dataset can be found in `election_2024_red_team.csv`

Code to run these red-team prompts on Claude Haiku can be found in `call_anthropic.py`

`medium_templates.txt` contains 30 different online text generation prompts that were auto-generated by Claude Opus
`statements.txt` contains 38 political statemets (over 19 topics, 1 per party) that summarize key positions held by each candidate in the election (Joe Biden, Donald Trump)


Analysis results on Haiku to come soon!
