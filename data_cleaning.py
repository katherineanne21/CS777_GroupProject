from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.sql.functions import col, sum
from pyspark.ml import Pipeline


def cleaning_flight_data(df):

    #selecting only these features, not selecting: ARR_DEL15 as (this leaks answer) or DEP_DELAY
    df = df.select(
        "MONTH",
        "DAY_OF_MONTH",
        "DAY_OF_WEEK",
        "OP_UNIQUE_CARRIER",
        "ORIGIN_AIRPORT_ID",
        "DEST_AIRPORT_ID",
        "AIR_TIME",
        "DISTANCE",
        "DEP_DEL15"
    )
    
    df.show(2)
    
    #checking if any column as nulls    
    df.select([
        sum(col(c).isNull().cast("int")).alias(c) for c in df.columns
    ]).show()
    
    #removing missing values
    df = df.dropna(subset=[
        "DEP_DEL15",
        "AIR_TIME",
        "DISTANCE",
        "OP_UNIQUE_CARRIER",
        "ORIGIN_AIRPORT_ID",
        "DEST_AIRPORT_ID"
    ])

    df = df.filter((df.DEP_DEL15 == 0) | (df.DEP_DEL15 == 1))  #making sure that dep_del15 is just 0 and 1
    df.show()
    
    return df

#Feature Engineering

def feature_eng():
    #converting categorical to numeric index
    carrier_indexer = StringIndexer(
        inputCol="OP_UNIQUE_CARRIER",
        outputCol="carrier_index"
    )
    
    origin_indexer = StringIndexer(
        inputCol="ORIGIN_AIRPORT_ID",
        outputCol="origin_index"
    )
    
    dest_indexer = StringIndexer(
        inputCol="DEST_AIRPORT_ID",
        outputCol="dest_index"
    )
    
    #one hot encoding
    #this creates numbers to vectors
    encoder = OneHotEncoder(
        inputCols=["carrier_index", "origin_index", "dest_index"],
        outputCols=["carrier_vec", "origin_vec", "dest_vec"]
    )
    
    #scaling only distance and time
    #combining them first
    num_assembler = VectorAssembler(
        inputCols=["DISTANCE", "AIR_TIME"],
        outputCol="num_features"
    )
    
    #scaling them
    scaler = StandardScaler(
        inputCol="num_features",
        outputCol="scaled_num_features"
    )
    
    #final feature vector
    final_assembler = VectorAssembler(
        inputCols=[
            "MONTH",
            "DAY_OF_MONTH",
            "DAY_OF_WEEK",
            "scaled_num_features",
            "carrier_vec",
            "origin_vec",
            "dest_vec"
        ],
        outputCol="features"
    )
    
    pipeline = Pipeline(stages=[
        carrier_indexer,
        origin_indexer,
        dest_indexer,
        encoder,
        num_assembler,
        scaler,
        final_assembler
    ])
    
    return pipeline
