# Articles

- [Medium](https://gordicaleksa.medium.com/how-i-got-a-job-at-deepmind-as-a-research-engineer-without-a-machine-learning-degree-1a45f2a781de) article about the story of [Aleksa Gordic](https://www.linkedin.com/in/aleksagordic?miniProfileUrn=urn%3Ali%3Afs_miniProfile%3AACoAACYdwKsB9_xmm5toYADSzYyGuIusSRinIsQ&lipi=urn%3Ali%3Apage%3Ad_flagship3_detail_base%3BJxNQXOXGTYCLEXUH4wJmJw%3D%3D)
- [Medium](https://towardsdatascience.com/my-journey-to-kaggle-master-at-the-age-of-14-e2c42b19c6f7) article on Andy Wang's (Kaggle competition master) suggestion on competing on kaggle
- [Kaggle](https://www.kaggle.com/tanulsingh077/tackling-any-kaggle-competition-the-noob-s-way) article by [Tanul singh](https://www.kaggle.com/tanulsingh077) (Kaggle competition master) on competing on kaggle.
- [SWE in ML at Google](https://leetcode.com/discuss/interview-experience/939742/Google-or-PayPal-or-SWE-or-ML-or-Nov-2020-or-Offer) with 1.5 YOE

<details>
  <summary>MLE Prep from <a href="https://www.teamblind.com/post/MLE---Interview-preparation-that-led-to-successful-offers-3k2aZ8d8" target="_blank">Blind </a></summary>
  ML Design / ML

* What I liked and what I recommend:

- http://www.mlebook.com/wiki/doku.php
Machine Learning Engineering by Andriy Burkov
This is a great resource. I read the entire book and I recommend doing the same. It provides a great overview of theory and practice for MLE. Being strong in aspects such as data drift, model monitoring, etc are essential.

- https://huyenchip.com/machine-learning-systems-design/toc.html
https://huyenchip.com/ml-interviews-book/

These materials authored by Chip Huyen are quite nice and give a reasonably good take on variety of topics. I found them to be a bit more shallow than the previous source, but overall good materials to follow.

- http://patrickhalina.com/posts/ml-systems-design-interview-guide/
ML Systems Design Interview Guide by Patrick Halina

This is one of the best resources in a very compact format. I used a very similar design process/format as the author describes in this post. ML design questions are really very streamlined and can be approached in a very systematic way. The author here introduces a great framework on how to solve them successfully. He introduces a perfect framework, in my view, to decompose these problems into smaller areas, and execute in these.

- https://developers.google.com/machine-learning/guides/rules-of-ml
The Rules of ML by Martin Zinkevich

Great resource that talks about very useful aspects such as when to start with heuristics, watching for silent failures, etc.

* What I have mixed feelings about:

- https://www.educative.io/courses/machine-learning-system-design
I tried this course, but it was kind of meh. It gives a good overview on how to approach the ML design problems (candidate generation + ranking in most of them), but it does not go deep in any of the areas. My recommendation is if you are very fresh and really need a lot of guidance, it might be worth it. Otherwise, the ones I included above are enough.

* What else to read:
- embedding: I found it essential to have a crisp and perfect understanding of embedding aspects. Word2vec, skip gram vs cbow, negative sampling, etc.
- A/B, model/data shift: This is something that often comes up a lot. You get extra points if you bring these topics by yourself (shift). MLE book above (1st position) gives a great overview on these topics.
- Basic algorithms: Linear/Logistic regression, Trees (bagging/boosting), SVM, Naive Bayes, deep learning. Knowing how they work and knowing when to choose which if it's preferred. Be prepared to provide side-by-side comparison any of the two.
- Model debugging: under/overfitting, regularization, etc. These are must. Some phone screens involved flash questions on these topics.
- Problem formulation. Practice and be very clear on formulating a problem as a machine learning problem. Be very clear about it.

* A few tips:
- When you are driving the design, check with interviewer often whether they would like to jump to section you are about to jump. I found all interviewers to be very open and allowing me to drive my way. Also, when you ask this question, if interviewer is missing something, they will probably ask for it.
- Similar to generic system design, never ever dive into the actual design right away. Clarification in the beginning is the key. Formulate the problem, discuss the requirements and expectations. Patrick's material above provides a great overview on this.
- Interviewers are looking at breadth and depth. Don't just say something, and immediately jump into details and go very deep. If interviewer will be interested in something specific, they will probably ask. Give alternatives and give your reasoning on trade-offs. Interviewers love discussing trade offs.
- Be positive: treat it like a discussion with a teammate on something you are working on, but you are driving the design. Don't just say things to impress the interviewer in any way. This will usually not work, and interviewers will spot this behavior right away.

---
System Design

I only did Grokking System Design and read DDIA. I found this to be enough, and not all of the places ask for generic design.
</details>


<hr>

# Other popular topics
1. Pseudo Labeling:
Build a high accurate model, predict on test set. For predictions with a high probability (say >90%), add those datapoints along with the predictions to the train set, and train again. <strong>Do not touch validation set.</strong>                              
Read: [here](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/52557#299825)
