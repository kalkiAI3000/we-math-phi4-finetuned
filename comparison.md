# Comparison: Base Phi-4 MM vs Finetuned We-Math Phi-4

Data dir: `/data-mount-large/data/wemath_phi4_processed`  
Base model: `/data-mount-large/models/Phi-4-multimodal-instruct`  
Finetuned model (HF): `kalkiai3000/we-math-phi4`

## Summary

- Samples compared: 100
- Base accuracy: 0.42 (42/100)
- Finetuned accuracy: 0.45 (45/100)
- FT wins (base fails, FT passes): 18

## Finetuned Wins (base incorrect, FT correct)

| # | is_mcq | image | ground_truth | base_pred | ft_pred |
|---|:-----:|:-----:|:------------:|:--------:|:------:|
| 21 | True | 2fa034c979df420faf16061a4f59f31d.jpg | b | a | b |
| 27 | False | 4432c8c40578478bab601c47420f5498.jpg | rectangle | circle | rectangle |
| 29 | False | 791ce9e0e41747e6bd4d7de4dd5a7175.jpg | triangle | cone | triangle |
| 32 | True | a391ff06cee84acdaaffd4345967919c.jpg | a | b | a |
| 41 | True | ebf44a7f80c8411fa949ad0931db2885.jpg | c | b | c |
| 45 | True | eb37d8d7cb204b7db35a657370ee4096.jpg | a | b | a |
| 51 | True | ab0b4e5a1dc84eaf9effcc901e862596.jpg | d | a | d |
| 52 | True | 2cabb23c9697434588b1d5131f5ae548.jpg | d | a | d |
| 68 | False | 0b89e61756dd4f8b80107e1802aee406.jpg | b | 4 | b |
| 69 | False | 236cee10df244036acda5cceec25a72a.jpg | i | d | i |
| 73 | False | 803043c2a6344cccba4aca90a4dcb7eb.jpg | b | 12.5 | b |
| 78 | False | eeb8afc97afb4556a54c4efa927cee29.jpg | b | c | b |
| 80 | False | 3d0ba4082a4e40a591c103a4f3fdb47d.jpg | b | c | b |
| 82 | True | e30726a8f52d4b249d33ca3b7101edf1.jpg | c | b | c |
| 86 | False | 260a18ca483742708f53618014a938e0.jpg | b | a | b |
| 91 | True | ea8015f270de4cd9930632db4230fb26.jpg | d | c | d |
| 97 | False | 1fd5afeeb1bd4121b729df08cf5f606b.jpg | 0.3 | 0.2 | 0.3 |
| 98 | False | 377b510a0e044b42a1179b54ad4678e3.jpg | 0.5 | 2.5 | 0.5 |

### Detailed examples (first 5 wins)

- Image: `2fa034c979df420faf16061a4f59f31d.jpg`
  - Question: fter rotating the triangle in the figure $180^\circ$ clockwise around the midpoint of its longest side and joining it with the original triangle, what is the new plane figure formed? A. Triangle; B. Parallelogram; C. Circle; D. Rhombus
  - Ground truth: `b`
  - Base prediction: `a`
  - Finetuned prediction: `b`

- Image: `4432c8c40578478bab601c47420f5498.jpg`
  - Question: hat is the shape of the front view of this figure?
  - Ground truth: `rectangle`
  - Base prediction: `circle`
  - Finetuned prediction: `rectangle`

- Image: `791ce9e0e41747e6bd4d7de4dd5a7175.jpg`
  - Question: hat is the shape of the side view of this figure?
  - Ground truth: `triangle`
  - Base prediction: `cone`
  - Finetuned prediction: `triangle`

- Image: `a391ff06cee84acdaaffd4345967919c.jpg`
  - Question: he figure shown is composed of six small squares. Can this figure be folded into a cube? A. Yes; B. No; C. Cannot answer
  - Ground truth: `a`
  - Base prediction: `b`
  - Finetuned prediction: `a`

- Image: `ebf44a7f80c8411fa949ad0931db2885.jpg`
  - Question: ompare the function values at the intersection points of line $l$ and function $C_{1}C_{2}$. A. $y_{1} > y_{2}$ B. $y_{1} = y_{2}$ C. $y_{1} < y_{2}$ D. Cannot be compared
  - Ground truth: `c`
  - Base prediction: `b`
  - Finetuned prediction: `c`
