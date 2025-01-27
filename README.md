# pine-explainer

PINE(Pair INterpretation for Entity matching) is an explainable entity matching algorithm.
PINE takes two
records as input, and outputs correlated token pairs as an explanation for an entity-matching decision. Our extensive experiments
on public datasets demonstrate that the extracted token pairs exhibit strong correlations and serve as interpretable evidence for matching records.

## Input and Output

- Input
  - Two entities(records) and entity matching model
- Output
  - Token pairs and the attribution scores

### Input Example

#### Record_left

| title | category | brand | modelno | price |
| ----- | -------- |------ | ------- | ---- |
| lexmark 1382920 toner 3000 page-yield black s | tationery & office machinery | lexmark | 1382920 |270.9 |

#### Record_right

| title | category | brand | modelno | price |
| ------|----------|-------|---------|------ |
| new-1382920 toner 3000 page yield black case pack 1 - 516495 | computer accessories | lexmark | | 574.19 |

#### Entity matching model

Arbitrary entity matching model which is wraped PINE's function call interface

### Output Example

| Token_left | Token_right | Attribution score |
| -----------|-------------|------------------ |
| 1382920    | new-1382920 | 0.739 |
| page-yield | yield       | 0.345 |
| 3000       | 3000        | 0.332 |
| toner      | toner       | 0.218 |
| machinery  | computer    | 0.036 |

## How to use

Please see [examples/PINE_example.ipynb](examples/PINE_example.ipynb)

