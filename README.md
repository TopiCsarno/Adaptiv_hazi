# N dimenziós neurális hálók - házi feladat
---

## Készítette: 
**Csarnó Tamás Péter** (F35OUX)

## Project
A házi feladat célja egy N dimenziós konvolúciós réteg, és N dimenziós pooling réteg megalkotása, amelyek segítségével magasabb dimenziójú adat konvolúciós feldolgozása is lehetséges. A ma elterjedt gépi tanulási keretrendszerek mint a Keras, jelenleg 3D-ig támogatják a konvolúciót.

## Elvégzett feladatok
- Irodalomkutatás
- Gépi tanulási model implementálása alacsony szintről (pythonban), ami képes tanító adatok alapján súlyokat módosítani hibavisszaterjesztéssel.
- Teljesen összekapcsolt réteg implementálása
- N dimenziós konvolúciós réteg megértése, kimenet képzés és gradiens számítás matematikájának levezetése és lekódolása
- Tesztelés
- Három gépi tanulási feladaton keresztül bemutatni a model működését.

## Fejlesztési lehetőség
- padding támogatása 
- tetszőleges minta alapján történő konvolúció
- hibakezelés nem megfelelő paraméterek esetén

## File struktúra

### Notebookok
Három szemléltető példán kereszül mutatom be az elkészített neurális háló rétegek működését. Ezek jupyter notebook formátumban (*.ipynb) találhatóak.

### /src
Forráskódot tartalmazza. Az elkészített neurális háló rétegek pedig a /src/layers könyvtárban találhatóak.

### /tests
Manuális teszteket tartalmaz amiket fejlesztés során használtam fel.

### /data
A [CNN_1D_classifier.ipynb](CNN_1D_classifier.ipynb) notebookban felhasznált előfeldolgozott adathalmazt tartalmazza. Az adathalmaz eredeti formájában letölthető innen: [LINK](https://paperswithcode.com/dataset/har) [1].

### /doc
A félév végi beszámolón bemutatott előadást tartalmazza.

## Referenciák
Irodalomkutatás során felhasznált publikációk:

[1] Anguita, D., Ghio, A., Oneto, L., Parra, X., & Reyes-Ortiz, J. L. (2012, December). Human activity recognition on smartphones using a multiclass hardware-friendly support vector machine. In International workshop on ambient assisted living (pp. 216-223). Springer, Berlin, Heidelberg.

[2] Jing, L., Wang, T., Zhao, M., & Wang, P. (2017). An adaptive multi-sensor data fusion method based on deep convolutional neural networks for fault diagnosis of planetary gearbox. Sensors, 17(2), 414.

[3] Zan, T., Wang, H., Wang, M., Liu, Z., & Gao, X. (2019). Application of multi-dimension input convolutional neural network in fault diagnosis of rolling bearings. Applied Sciences, 9(13), 2690.
