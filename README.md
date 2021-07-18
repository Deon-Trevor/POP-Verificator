# [Proof of Payment Verificator](https://github.com/Deon-Trevor/POP-Verificator)

<a href="https://github.com/Deon-Trevor/POP-Verificator">

![](https://github.com/Deon-Trevor/Github-Stats/blob/master/generated/overview.svg)
![](https://github.com/Deon-Trevor/Github-Stats/blob/master/generated/languages.svg)

</a>

This is a machine learning powered proof of payment verification script. It utilizes Object Character Recognition to 
detect objects on a document. Once detected, text is extracted from the documents and used to train the model to classify document
types. Using the text, information is extracted and used to validate the Proof of Payment document.

It can also detect rotated images, realign and read from them. So far the rotation is possible for images and single-page pdfs. Multiple-page pdf rotation detection is still under development.

The script can also be served as Flusk API to make it an easy plug and play to your pipeline.