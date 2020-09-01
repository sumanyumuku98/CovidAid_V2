# CovidAID for Detection of SARS-CoV-2 from CXR using Attention Guided CNN.

This is an extention of work done by [arpanmangal](https://github.com/arpanmangal/CovidAID). The model takes a CXR as an input and outputs the probability scores for 3 classes (`NORMAL`, `Pneumonia` and `COVID-19`) or 4 classes (`Normal`, `Bacterial Pneumonia`, `Viral Pneumonia`, `Covid-19`)

It is based on [Diagnose like a Radiologist: Attention Guided Convolutional Neural Network for Thorax Disease Classification](https://arxiv.org/abs/1801.09927) and its reimplementation by [Ien001](https://github.com/Ien001/AG-CNN). The initial weights used for training were obtained from [arnoweng](https://github.com/arnoweng/CheXNet).

## Dataset
`CovidAID` uses the [covid-chestxray-dataset](https://github.com/ieee8023/covid-chestxray-dataset), [BSTI-dataset](https://www.bsti.org.uk/training-and-education/covid-19-bsti-imaging-database/)for COVID-19 X-Ray images and [chest-xray-pneumonia](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) dataset for data on Pneumonia and Normal lung X-Ray images. 


## Getting Started


## Results

We present the results in terms of both the per-class AUROC (Area under ROC curve) on the lines of `CheXNet`, as well as confusion matrix formed by treating the most confident class prediction as the final prediction. We obtain a mean AUROC of `0.9738` (4-class configuration).

<center>
<table>
<tr><th></th><th>3-Class Classification</th></tr>
<tr>
<td></td>
<td>

| Pathology  |   AUROC    | Sensitivity | PPV
| :--------: | :--------: | :--------: | :--------: |
| Normal Lung  | 0.9795 | 0.744 | 0.989
| Bacterial Pneumonia | 0.9814 | 0.995 | 0.868
| COVID-19 | 0.9997 | 1.000 | 0.968

</td></tr> 
<tr>
<td>ROC curve</td>
<td>

![ROC curve](./assets/roc_3.png "ROC curve")

</td>
</tr>
<tr>
<td>Confusion Matrix</td>
<td>

![Normalized Confusion Matrix](./assets/cm_3.png "Normalized Confusion Matrix")

</td>
</tr>



</table>
</center>

## Visualizations
To  demonstrate  the  results  qualitatively,  we  generate  saliency  maps  for  our model’s  predictions  using  RISE. The purpose of these visualizations was to have an additional check to rule out  model  over-fitting  as  well  as  to  validate  whether  the  regions  of  attention correspond to the right features from a radiologist’s perspective. Below are some of the saliency maps on COVID-19 positive X-rays.

<center>

<table>
<tr>
<td>

![Original 1](./assets/visualizations/original_1.png "Original 1") 

</td><td> 

![Original 2](./assets/visualizations/original_2.png "Original 2") 

</td><td> 

![Original 3](./assets/visualizations/original_3.png "Original 3")

</td></tr>

<tr><td> 

![Visualization 1](./assets/visualizations/vis_1.png "Visualization 1") 

</td><td> 

![Visualization 2](./assets/visualizations/vis_2.png "Visualization 2") 

</td><td>

![Visualization 3](./assets/visualizations/vis_3.png "Visualization 3")

</td></tr>
</table>


</center>

