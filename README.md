Input: Researcher describes the research idea (e.g., “I want to compare Transformer and Mamba on long-context summarization”).


Planning: AI figures out:


Which benchmarks are relevant


Which experiments to run


Which metrics to collect


Implementation: AI writes code for each experiment.


Execution: Code is run.


Analysis: Results are collected and summarized.


Output: Report + ready-to-share code.



Why Full Scope is Hard in 5 Hours
Multi-domain expertise: Benchmark discovery, experiment planning, and code generation for multiple frameworks is a lot of moving parts.


Running time: Real ML benchmarks can take hours/days.


Infra complexity: Need datasets, GPU access, and code environment automation.



How to Scope It for a Hackathon
You can fake or shortcut the heavy parts so the demo still feels magical without actually training huge models.
 Here’s a scoped, 5-hour hackathon-friendly version:

Refined Idea: “Auto-Experiment Designer & Runner” (Prototype)
Input: Researcher describes an experiment.


AI Planning:


Identify benchmark datasets (from a small, curated list you preload).


Select metrics.


Propose experimental setup.


Code Generation:


Use templates + LLM to fill in dataset/model-specific details.


Support just 2–3 preloaded ML tasks (e.g., text classification, image classification, regression).


Execution:


Instead of hours of training, run tiny fast models (logistic regression, small CNN) on toy subsets.


Output:


Results table + AI-written summary (e.g., “Model A outperforms Model B by 12% on Accuracy, but loses on F1”).


Generated README.md with code + results.



Example Demo Flow
Judge says:
 “I want to compare a CNN and a ResNet on CIFAR-10 accuracy.”


AI outputs:


Selected benchmark: CIFAR-10


Metrics: Accuracy, F1


Experimental plan: Train both models for 3 epochs.


Code auto-generates, runs on CPU in ~1–2 minutes.


System produces:


Plot comparing results.


Written summary: “ResNet achieved 74.3% accuracy, 12% higher than CNN, but required 3× more parameters.”


GitHub-ready code folder.



Technical Shortcut Ideas
Preload datasets (MNIST, CIFAR-10, IMDB sentiment).


Use small models so training is seconds/minutes.


Cache results to avoid reruns.


Template code for PyTorch/Scikit-learn and just swap dataset/model names.


Use LLM for:


Natural language → experiment config


Code templating


Result interpretation



Why This Works for Judges
Looks like a miniature “AutoML for researchers” system.


Combines LLM reasoning + tool use + code execution.


Output is tangible: code, results, and analysis in one go.


Feels like something you could scale to real research with more time.



If you want, I can give you a 5-hour build plan that gets you from blank repo to demo-ready with this scoped “Auto-Experiment Designer & Runner” — so you can walk in and impress judges without frying your GPU.
Do you want me to draft that plan?

