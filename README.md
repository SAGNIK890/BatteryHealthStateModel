# Battery Health State Classification using Machine Learning

## What this project is about

This project is about predicting the health condition of a lithium-ion battery from its discharge behaviour.

Instead of treating battery health as just a single number, I divided the battery condition into three practical classes:

- Healthy — SoH >= 85%
- Degrading — 75% <= SoH < 85%
- Critical — SoH < 75%

The main idea was to take the information already present in the NASA battery cycling data and turn each discharge cycle into a set of useful features. I then used those features to classify the battery into one of the three health states.

The interesting part of this project was not just getting a model to work. I initially used a Random Forest, which gave an accuracy of 78.5%. I then looked closely at where the model was making mistakes instead of simply accepting that number. That led me to try a Decision Tree, and the resulting confusion matrix shows an accuracy of 89.5%.

So the important part of the project is really the process:

raw battery data → cleaning → feature extraction → SoH calculation → health classes → Random Forest → error analysis → Decision Tree → improved classification

# 1. Dataset

The project uses the NASA battery dataset obtained through the patrickfleith/nasa-battery-dataset Kaggle dataset.

The metadata contains charge, discharge and impedance records. For the actual health-classification model, I focused on discharge cycles, because discharge behaviour gives us useful information about how much usable capacity the battery can actually deliver.

The original metadata contains 7,565 records, of which:

- 2,815 are charge records
- 2,794 are discharge records
- 1,956 are impedance records

After selecting discharge cycles, there were 2,794 discharge records from 34 batteries.

This grouping by battery is important. A battery has many cycles, so randomly mixing cycles from the same battery into training and testing can make the model look better than it actually is. I therefore used a group-based split using battery_id, so that the test set contains batteries that were not used for training.

# 2. Turning raw discharge data into useful information

The raw data contains measurements such as:

- Voltage
- Current
- Temperature
- Time

For every discharge cycle, I extracted summary features rather than feeding every individual measurement directly into the classifier.

The main features were:

| Feature | What it represents |
|---|---|
| cycle | The position of the cycle in the battery's life |
| ambient_temperature | Temperature around the battery during the test |
| v_mean | Mean measured voltage during discharge |
| v_min | Minimum measured voltage |
| v_max | Maximum measured voltage |
| v_std | Variation in voltage during discharge |
| i_mean | Mean discharge current |
| temp_mean | Mean battery temperature |
| temp_max | Maximum temperature reached |
| temp_rise | Temperature increase during discharge |
| discharge_time_s | Total discharge duration |
| time_to_3_5V_s | Time taken for the voltage to reach 3.5 V |

I chose these because battery degradation is reflected in several physical behaviours at the same time.

Capacity fade, voltage behaviour, temperature behaviour and discharge duration are all connected to the condition of the battery.

For example, if a battery reaches the lower voltage region much earlier than it used to, that is useful information about its condition. This is why time_to_3_5V_s is more meaningful than simply looking at one voltage measurement.

# 3. Cleaning the data

The first feature table contained 2,794 rows and 14 columns.

Before training, I removed problematic records by:

- removing infinite values
- removing missing rows where necessary
- removing duplicate rows
- keeping realistic capacity values
- removing discharge cycles that were too short

After cleaning, 2,562 usable discharge cycles remained.

The health distribution was:

| Health state | Number of cycles |
|---|---:|
| Healthy | 1,638 |
| Degrading | 658 |
| Critical | 266 |
| Total | 2,562 |

This distribution is important when interpreting accuracy.

The classes are not equally represented. Healthy batteries make up the largest part of the dataset, while Critical is the smallest class.

Because of that imbalance, accuracy alone is not enough to judge the model.

# 4. How State of Health was calculated

I calculated State of Health relative to the initial capacity of each battery.

For each battery:

```text
SoH = Current Capacity / Initial Capacity
```

The initial capacity was estimated from the median capacity of the first three usable cycles.

So if a battery initially had approximately 2 Ah and later had 1.6 Ah:

```text
SoH = 1.6 / 2.0
     = 0.80
     = 80%
```

That battery would fall into the Degrading class.

The thresholds used in the project were:

```text
SoH >= 85%        → Healthy
75% <= SoH < 85% → Degrading
SoH < 75%        → Critical
```

This converts a continuous battery-health value into a classification problem.

# 5. Why I started with Random Forest

I started with Random Forest because it is a strong baseline for tabular data.

Random Forest does not depend on one decision tree. It builds many trees and combines their predictions. That usually makes it less sensitive to the peculiarities of a single tree.

The Random Forest used in the initial experiment had:

```python
RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
```

I also used class_weight="balanced" because the three health classes were not equally represented.

The model was trained using 1,962 samples and evaluated on 600 samples from the held-out batteries.

The first result was:

```text
Accuracy = 78.5%
```

At first glance, 78.5% looks reasonable.

But the classification report showed why the number was not telling the complete story.

## Random Forest classification report

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Healthy | 0.78 | 0.96 | 0.86 | 394 |
| Degrading | 0.88 | 0.49 | 0.63 | 181 |
| Critical | 0.36 | 0.20 | 0.26 | 25 |
| Accuracy | | | 0.79 | 600 |
| Macro average | 0.67 | 0.55 | 0.58 | 600 |
| Weighted average | 0.79 | 0.79 | 0.76 | 600 |

The most important thing I noticed here was not the 78.5% accuracy.

It was the recall.

For the Degrading class, recall was only 49%.

For Critical, recall was only 20%.

That means the model was missing a large proportion of the batteries that actually belonged to these two classes.

For a battery health system, that is something I did not want to ignore.

# 6. What the Random Forest was actually doing

The Random Forest was very good at recognising Healthy samples.

Healthy recall was 96%.

So most genuinely Healthy cycles were correctly identified.

But this also tells us something else.

The model was much less successful when the battery was moving away from the Healthy region.

The Degrading class had:

```text
Precision = 88%
Recall    = 49%
F1-score  = 63%
```

The high precision means that when the model said "Degrading", it was usually correct.

The low recall means it was not finding enough of the batteries that were actually Degrading.

The Critical class was even more difficult:

```text
Precision = 36%
Recall    = 20%
F1-score  = 26%
```

Only about one out of five actual Critical samples were being detected.

This is where looking only at accuracy can become misleading.

Because Healthy samples were the largest group, getting Healthy predictions right contributed a lot to the overall accuracy.

So I went back to the actual errors instead of treating 78.5% as the final answer.

# 7. What the model was learning from

The Random Forest feature importance also gave a useful clue about the problem.

The five most important features were:

| Feature | Importance |
|---|---:|
| i_mean | 0.184 |
| time_to_3_5V_s | 0.156 |
| v_std | 0.105 |
| cycle | 0.100 |
| temp_rise | 0.088 |

This made physical sense.

The model was relying heavily on:

- discharge current behaviour
- how quickly the battery reached 3.5 V
- voltage variation
- cycle count
- temperature rise

In other words, the classifier was not simply memorising one feature.

It was using several indicators of battery degradation.

# 8. Why I tried a Decision Tree

After seeing the Random Forest errors, I tried a simpler model: a Decision Tree.

The reason was not "Decision Trees are always better than Random Forests."

They are not.

The reason was more specific to this dataset.

The three classes are defined using clear SoH thresholds:

```text
>= 85%       → Healthy
75% to <85%  → Degrading
<75%         → Critical
```

Battery behaviour also contains a number of threshold-like relationships.

A Decision Tree is naturally good at representing this type of logic.

Conceptually, a tree can learn rules similar to:

```text
Is the discharge behaviour still consistent with a healthy battery?
        |
        +-- Yes → Healthy
        |
        +-- No
             |
             |-- Does the behaviour fall into the degradation region?
             |       |
             |       +-- Yes → Degrading
             |
             +-- No → Critical
```

The actual tree is more complicated than this, but this is the basic idea.

Another advantage was interpretability.

With a single Decision Tree, it is much easier to follow the sequence of decisions that leads to a prediction.

For a project where I wanted to understand why a battery was being placed into a particular health class, this was useful.

So the switch was driven by the error pattern of the first model, not by blindly replacing one algorithm with another.

# 9. The Decision Tree result

The stored confusion matrix in this repository is the Decision Tree result.

It reports:

```text
Accuracy = 89.5%
```

The test set still contains 600 samples.

The confusion matrix is:

| Actual \ Predicted | Healthy | Degrading | Critical |
|---|---:|---:|---:|
| Healthy | 380 | 14 | 0 |
| Degrading | 49 | 132 | 0 |
| Critical | 0 | 0 | 25 |

The diagonal values are the correct predictions:

```text
Healthy    → 380
Degrading  → 132
Critical   → 25
```

Therefore:

```text
Correct predictions = 380 + 132 + 25
                    = 537

Total predictions = 600

Accuracy = 537 / 600
         = 0.895
         = 89.5%
```

So the improvement from the original Random Forest accuracy was:

```text
89.5% - 78.5%
= 11.0 percentage points
```

This is a substantial change on the same 600-sample evaluation size.

# 10. Reading the Decision Tree confusion matrix

The easiest way to understand a confusion matrix is to remember:

- Rows = what the battery actually was
- Columns = what the model predicted

So the matrix:

```text
                 Predicted
              H     D     C

Actual H     380    14     0
Actual D      49   132     0
Actual C       0     0    25
```

needs to be read row by row.

## Healthy

There were:

```text
394 actual Healthy samples
```

The model correctly classified:

```text
380
```

and incorrectly classified:

```text
14
```

as Degrading.

It did not classify any Healthy sample as Critical.

Therefore:

```text
Healthy recall = 380 / 394
               ≈ 96.45%
```

So the model found almost all Healthy samples.

## Degrading

There were:

```text
181 actual Degrading samples
```

The model correctly found:

```text
132
```

But:

```text
49
```

were incorrectly predicted as Healthy.

Importantly, none of the Degrading samples were predicted as Critical.

Therefore:

```text
Degrading recall = 132 / 181
                 ≈ 72.93%
```

This is a major improvement compared with the Random Forest's 49% recall.

The model became much better at actually detecting the middle, Degrading state.

## Critical

There were:

```text
25 actual Critical samples
```

and the Decision Tree correctly identified:

```text
25
```

of them.

There were 0 Critical samples predicted as Healthy.

There were also 0 Critical samples predicted as Degrading.

So:

```text
Critical recall = 25 / 25
                = 100%
```

For this particular test split, every Critical sample was correctly identified.

This is one of the most noticeable differences from the Random Forest result, where Critical recall was only 20%.

# 11. Precision, recall and F1-score

Accuracy gives the overall percentage of correct predictions, but it does not tell me how each class behaves.

That is why I looked at precision, recall and F1-score.

## Precision

Precision answers:

"When the model predicts this class, how often is it actually correct?"

Formula:

```text
Precision = TP / (TP + FP)
```

### Healthy

```text
TP = 380
FP = 49

Precision = 380 / (380 + 49)
          ≈ 88.58%
```

So when the model predicts Healthy, about 88.6% of those predictions are actually Healthy.

### Degrading

```text
TP = 132
FP = 14

Precision = 132 / (132 + 14)
          ≈ 90.41%
```

So the model's Degrading predictions are quite precise.

### Critical

```text
TP = 25
FP = 0

Precision = 25 / 25
          = 100%
```

There were no false Critical predictions in this test set.

# 12. Recall

Recall answers:

"Out of all the samples that really belong to this class, how many did the model find?"

Formula:

```text
Recall = TP / (TP + FN)
```

### Healthy

```text
Recall = 380 / 394
       ≈ 96.45%
```

### Degrading

```text
Recall = 132 / 181
       ≈ 72.93%
```

### Critical

```text
Recall = 25 / 25
       = 100%
```

The recall numbers are particularly interesting because they show where the model improved.

The original Random Forest had:

```text
Healthy recall    = 96%
Degrading recall  = 49%
Critical recall   = 20%
```

The Decision Tree confusion matrix gives:

```text
Healthy recall    ≈ 96.45%
Degrading recall  ≈ 72.93%
Critical recall   = 100%
```

So the biggest change was not really in Healthy classification.

Healthy was already being detected well.

The big improvement happened in the lower-health classes.

# 13. F1-score

F1-score combines precision and recall.

Formula:

```text
F1 = 2 × (Precision × Recall)
        ----------------------
        (Precision + Recall)
```

It is useful when I want a balance between:

- not making too many false predictions
- not missing too many actual samples

For the Decision Tree:

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Healthy | 88.58% | 96.45% | 92.35% |
| Degrading | 90.41% | 72.93% | 80.73% |
| Critical | 100% | 100% | 100% |

The Degrading class has a lower F1-score than Healthy because its recall is still lower.

That is visible directly in the confusion matrix.

49 Degrading samples are still being predicted as Healthy.

So the Decision Tree did not magically solve every classification error.

It reduced the errors substantially, especially for the more degraded classes.

# 14. Macro average vs weighted average

These two averages are worth mentioning because the dataset is imbalanced.

## Macro average

Macro average gives every class equal importance.

For the Decision Tree, based on the confusion matrix:

```text
Macro precision ≈ 93.00%
Macro recall    ≈ 89.79%
Macro F1        ≈ 91.03%
```

This is useful because Critical has only 25 samples, but macro averaging still gives Critical the same weight as Healthy and Degrading.

## Weighted average

Weighted average takes the number of samples in each class into account.

Because Healthy has many more samples than Critical, Healthy contributes more to the weighted average.

From the confusion matrix:

```text
Weighted precision ≈ 89.61%
Weighted recall    = 89.50%
Weighted F1        ≈ 89.16%
```

This is why I would not report only one average.

The macro average tells me how the model behaves across classes more equally, while the weighted average reflects the actual class distribution in the test set.

# 15. What actually changed from 78.5% to 89.5%

The important change was not simply:

```text
Random Forest
        ↓
Decision Tree
        ↓
89.5%
```

The more useful way to look at it is:

```text
Random Forest
    |
    | 78.5% accuracy
    |
    | Look at classification report
    |
    | Healthy recall = 96%
    | Degrading recall = 49%
    | Critical recall = 20%
    |
    ↓
Identify the weak part of the classifier
    |
    | The lower-health classes are being missed
    |
    ↓
Try a simpler, threshold-oriented model
    |
    ↓
Decision Tree
    |
    | Healthy recall ≈ 96.45%
    | Degrading recall ≈ 72.93%
    | Critical recall = 100%
    |
    ↓
89.5% accuracy
```

So the improvement came mainly from reducing misclassification in the Degrading and Critical regions.

The Decision Tree still makes 49 Degrading → Healthy mistakes, but it completely separates the 25 Critical test samples in this particular split.

# 16. Why a Decision Tree can make sense for this problem

A battery does not suddenly become "Critical" because one sensor value crossed one magic number.

Battery degradation is a combination of changes in several signals.

However, the final health labels themselves are threshold-based:

```text
Healthy    >= 85%
Degrading  75% to <85%
Critical   <75%
```

A Decision Tree is good at building a sequence of threshold decisions.

For example, conceptually it can learn relationships such as:

```text
cycle count
     +
discharge duration
     +
voltage variation
     +
temperature rise
     +
current behaviour
     ↓
health state
```

This also makes the model easier to inspect than a large ensemble.

The point is not that Decision Trees are universally better than Random Forests.

The result here is specific to this experiment and this particular test split.

# 17. A very important limitation

There is one thing I would not hide in this project.

The reported 89.5% is the result shown by the Decision Tree confusion matrix stored in the repository.

The notebook currently contains the Random Forest training/evaluation code that reports 78.5%, while the stored confusion-matrix image is labelled:

"Decision Tree - confusion matrix (accuracy = 89.5%)"

So the repository documents the transition through its artifacts, but the exact Decision Tree training cell/hyperparameters are not currently represented in the same notebook section as the Random Forest experiment.

Because of that, I would describe the 89.5% result as the recorded Decision Tree result for the 600-sample evaluation set, rather than claiming that the current notebook itself contains a complete reproducible Decision Tree experiment.

This distinction matters if someone else wants to reproduce the result exactly.

# 18. What I learned from the confusion matrix

The biggest lesson from this project was that accuracy alone can hide the actual problem.

The Random Forest gave:

78.5% accuracy

but its Critical recall was only:

20%

The Decision Tree result gave:

89.5% accuracy

and the stored confusion matrix shows:

25 / 25 Critical samples correctly classified

At the same time, the model still has room for improvement because:

```text
49 Degrading samples
        ↓
were classified as Healthy
```

That is the main remaining error pattern.

So if I continue this project, I would focus specifically on separating:

Healthy vs Degrading

rather than simply trying random algorithms until the accuracy goes up.

# 19. Project files

The repository contains:

```text
BatteryHealthStateModel/
│
├── Module1.ipynb
├── README.md
├── cleaned_battery_dataset.csv
├── battery_health_model.joblib
├── confusion_matrix.png
├── chart1_voltage_vs_cycle.png
├── chart2_health_distribution.png
├── chart3_capacity_fade.png
├── Sagnik_Bhattacharyya_Module-1.pdf
└── Sagnik_Bhattacharyya_Module-1.pptx
```

The notebook handles the main data-processing and Random Forest experiment, while the saved files provide the processed dataset, trained model and visual outputs.

# 20. Main takeaways

## Data

- Started with NASA battery cycling data.
- Focused on discharge cycles.
- Used 34 batteries.
- Extracted physical and temporal features from each discharge cycle.
- Cleaned the data down to 2,562 usable cycles.

## Health definition

```text
SoH >= 85%        → Healthy
75% <= SoH < 85% → Degrading
SoH < 75%        → Critical
```

## First model

Random Forest:

```text
Accuracy = 78.5%
```

The major weakness was detecting degraded batteries:

```text
Degrading recall = 49%
Critical recall  = 20%
```

## Second model

Decision Tree:

```text
Accuracy = 89.5%
```

From the stored confusion matrix:

```text
Healthy recall    ≈ 96.45%
Degrading recall  ≈ 72.93%
Critical recall   = 100%
```

The improvement was therefore not just about getting a larger accuracy number.

The confusion matrix shows that the classifier became much better at distinguishing the lower-health states in this evaluation set.

# Conclusion

The main thing I wanted to achieve with this project was to move from raw battery measurements to something that can actually describe the condition of a battery.

The first Random Forest gave me a baseline of 78.5%.

Instead of stopping there, I checked the classification report and found that the model was particularly weak at identifying Degrading and Critical batteries.

That made me question whether a more complicated ensemble was actually the best fit for the way the classes were defined.

I then moved to a Decision Tree.

The stored result reached 89.5% accuracy, with the confusion matrix showing 537 correct predictions out of 600.

More importantly, the matrix makes the improvement understandable.

The model correctly identified all 25 Critical samples in the test set and improved the detection of Degrading samples substantially compared with the Random Forest baseline.

There are still mistakes, especially the 49 Degrading samples classified as Healthy, so I would not consider the problem completely solved.

But this experiment showed me something useful:

"Looking at where a model fails can be more informative than simply looking at its overall accuracy."

That was the main reason for moving from Random Forest to Decision Tree in this project.
