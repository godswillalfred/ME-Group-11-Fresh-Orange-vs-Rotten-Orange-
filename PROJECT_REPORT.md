# Brief Project Report

Group ME11 developed a binary image classifier for distinguishing fresh oranges
from rotten oranges. Images were obtained from the Fruits Fresh and Rotten for
Classification dataset on Kaggle. The orange classes were selected, checked for
invalid and duplicate files, and divided into training, validation and test
sets. A MobileNetV3Small transfer-learning model was trained and evaluated in
Google Colab before being connected to a Streamlit web application. To use the
application, the user uploads a clear orange image, presses **Analyse Orange**,
and receives the predicted condition with a confidence score. The main
challenges were variations in lighting, backgrounds, stages of decay and the
binary model's tendency to classify unsupported images. Data augmentation,
early stopping, threshold selection and an uncertainty warning reduced these
problems. 
  
**GitHub repository:**ME-Group-11-Fresh-Orange-vs-Rotten-Orange
**Deployed application:**https://me-group11-fresh-orange-vs-rotten-orange.streamlit.app/

# Group ME11 Member Details

Complete this table with every member who genuinely participated.

| Name | Registration number | GitHub username | Contribution |
|---|---|---|---|
| Bassey, God'swill Alfred| 22/EG/ME/1765| godswillalfred |Team leader,project coordination, GitHub repository creation, deployment to streqamlit cloud |
| Ekpenyong, Abasifreke Francis| 22/EG/ME/1745| abeefree44-hash| download of dataset  |
| Daniel, Allwell Sylvanus| 22/EG/ME/1735| allwelldaniel24 | download of images  |
| Eyo, Gideon Monday| 22/EG/ME/1705| eyogideon2003-cmd | code preview  |
| Christopher, Ubong Victor | 22/EG/ME/1755| christopherubong15-stack | model evalution  |
| Egan, Victor Agbor| 22/EG/ME/1795| victoregan | code preview  |
| Uzoma, Ephraim Anointing| 22/EG/ME/1695| ephyjack | Testing    |
| Eluwah Divine Ikenna| 22/EG/ME/1745| eluwahdivine | Report preparation |
| Anietie, Fortune Ekong| 22/EG/ME/1725| anietiefortune03-oss | code preview |
| Thompson, Paul Sunday| 22/EG/ME/1825| pauldeapostle | code Review  |
| Francis Edwin Edwin| 22/EG/ME/1775| francisedwin784-sketch | code review  |
| Christian Emmanuel Elisha| 22/EG/ME/1715| eelisha116-beep | code preview  |
