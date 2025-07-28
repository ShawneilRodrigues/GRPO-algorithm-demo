Understanding Group-wise Policy Optimization (GRPO)
This document provides a technical explanation of the Group-wise Policy Optimization (GRPO) algorithm, a method for aligning Large Language Models (LLMs) with human preferences or specific objectives defined by a reward function. We will break down the core equations and explain how GRPO works in practice.

1. The Core Objective of GRPO
Like other alignment algorithms (e.g., DPO, PPO), the goal of GRPO is to fine-tune a base LLM (the reference policy, π_ref) to create a new model (the learned policy, π_θ) that maximizes a given reward function r(x, y) while not deviating too far from the original model's capabilities.

The theoretical objective that GRPO aims to maximize is:

L 
GRPO
​
 (π 
θ
​
 )=E 
x∼D,y∼π 
θ
​
 (⋅∣x)
​
 [r(x,y)]−β⋅E 
x∼D
​
 [KL(π 
θ
​
 (⋅∣x)∣∣π 
ref
​
 (⋅∣x))]
Let's break this down:

π_θ(y|x): The probability of the new, fine-tuned model (our policy) generating response y given prompt x.

π_ref(y|x): The probability of the original, pre-trained model (the reference policy) generating response y.

r(x, y): The reward function. This is a scalar value that scores how "good" a response y is for a prompt x. A higher score is better.

β: The KL-divergence coefficient. This is a hyperparameter that controls the strength of the penalty for deviating from the reference model.

KL(π_θ || π_ref): The Kullback-Leibler (KL) divergence. This measures how much the learned policy π_θ has "moved away" from the original reference policy π_ref.

In simple terms, the goal is to find a policy π_θ that gets high rewards, but we penalize it if it becomes too different from the original, capable LLM.

2. From Theory to Practice: The GRPO Loss Calculation
Since we cannot calculate the expectation over all possible responses, GRPO uses a sampling-based approach. For each prompt x in a training batch, the algorithm performs the following steps:

Step 1: Generate a Group of Responses
Instead of comparing just two responses (like in DPO), GRPO generates a group of K responses from the current policy π_θ:

{y 
1
​
 ,y 
2
​
 ,…,y 
K
​
 }∼π 
θ
​
 (⋅∣x)
Step 2: Calculate Rewards and Advantage
Each of the K responses is scored using the reward function r(x, y). This gives us a set of reward scores:

{r 
1
​
 ,r 
2
​
 ,…,r 
K
​
 }wherer 
i
​
 =r(x,y 
i
​
 )
In GRPO, the advantage of a response is implicitly defined by its reward relative to the other responses in the group. To turn these raw scores into a usable probability distribution, we apply the Softmax function. This gives higher weights to better responses.

weights 
i
​
 =softmax(r 
i
​
 )= 
∑ 
j=1
K
​
 e 
r 
j
​
 
 
e 
r 
i
​
 
 
​
 
A response with a high reward will get a high weight, while a response with a low reward will get a low weight.

Step 3: Calculate the Weighted Log-Likelihood Loss
The final step is to update the model π_θ to increase the probability of generating the high-reward responses. This is done by calculating a weighted log-likelihood loss.

First, we need the log-probability of each generated response y_i according to the current policy π_θ. This is derived directly from the model's logits (the raw, unnormalized scores the model outputs for each token). The log-probability log π_θ(y_i|x) is calculated by summing the log-softmax of the logits for each token in the response y_i. In practice, this is equivalent to the negative cross-entropy loss.

The GRPO loss is the negative of the weighted sum of these log-probabilities:

L 
GRPO
​
 =− 
i=1
∑
K
​
 weights 
i
​
 ⋅logπ 
θ
​
 (y 
i
​
 ∣x)
By minimizing this loss L, we are effectively maximizing the log-probabilities of the responses (log π_θ) that have high weights (i.e., high rewards).

3. Stabilizing Training: Clipping and KL Regularization
Training LLMs with reinforcement learning can be unstable. GRPO incorporates two key mechanisms from algorithms like PPO to ensure stable and effective training.

A. PPO-Style Clipping (Reducing Overfitting)
A major risk in policy optimization is that the model takes an update step that is too large, drastically changing its behavior and leading to a collapse in performance. This is a form of overfitting to a specific batch of data.

To prevent this, we can clip the policy update. We calculate the importance ratio between the new policy π_θ and the policy before the update, π_θ_old:

ratio(θ)= 
π 
θ 
old
​
 
​
 (y 
i
​
 ∣x)
π 
θ
​
 (y 
i
​
 ∣x)
​
 
This ratio tells us how much more (or less) likely the new policy is to produce a given response compared to the old one. If this ratio is very large, the update is too aggressive.

We then modify the loss objective to clip this ratio within a small range [1-ε, 1+ε] (e.g., ε = 0.2). The update for a single response becomes:

L 
clipped
​
 =min(ratio(θ)⋅A 
i
​
 ,clip(ratio(θ),1−ϵ,1+ϵ)⋅A 
i
​
 )
Where A_i is the advantage (our softmax-normalized reward weights_i). This ensures that the policy update for any single response cannot be excessively large, leading to more stable training.

B. The Role of Beta (β): Preventing Policy Deviation
The second stabilization mechanism is the KL-divergence penalty, controlled by the hyperparameter β.

KL Penalty=β⋅KL(π 
θ
​
 (⋅∣x)∣∣π 
ref
​
 (⋅∣x))
What it does: This term measures how different the output distribution of our fine-tuned model π_θ is from the original π_ref. If π_θ starts generating text that π_ref would find very unlikely, the KL divergence will be high.

Why it's important: The reward function is never perfect and can be "hacked." For example, a model might learn that repeating "Thank you" a hundred times yields a high reward. This is called reward hacking. The KL penalty prevents this by ensuring the model doesn't stray too far from the coherent, fluent, and knowledgeable behavior of the original base model.

How β works:

If β is high, the model is heavily penalized for deviating from the reference model. It will be very safe but may not learn the desired behavior as strongly.

If β is low, the model has more freedom to maximize the reward function, but it risks reward hacking and losing its general capabilities.

By combining the weighted log-likelihood loss with the KL penalty, the full objective ensures that the model learns to achieve high rewards while remaining a coherent and capable language model.
