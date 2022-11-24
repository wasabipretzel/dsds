## Data Science Project

+ 한양대학교 데이터 사이언스 프로젝트 수업

+ structure

```bash
.
|-- config
|   |-- base_conf.py
|   `-- readme.md
|-- data
|   `-- raw
|       |-- Dealer Hierarchies.csv
|       |-- Industry data.csv
|       |-- Product Hierarchies.csv
|       |-- Profit per Product.csv
|       `-- Retail data.csv
|-- docs
|   `-- readme.md
|-- func
|   `-- readme.md
|-- models
|   `-- readme.md
|-- notebooks
|   `-- DataScience project.ipynb
`-- output
    `-- readme.md
```


### folder usage
+ 전역변수, 경로는 ```config/base_conf.py``` 에 작성
+ 데이터는 raw폴더의 데이터를 사용하여 가공된 데이터는 따로 폴더를 생성하여 저장 후 path를 ```config/base_conf.py``` 에 등록
+ 전처리 등 자주 쓰이는 함수는 ```func/``` 에 작성
+ Modeling의 output은 ```output/``` 폴더에 
