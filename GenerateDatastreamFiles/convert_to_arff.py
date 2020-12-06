import argparse
import pandas as pd
import numpy as np

def saveStreamToArff(filename, stream_examples):
    print(filename)
    with open(f"{filename}", 'w') as f:
        f.write(f"@RELATION stream\n")
        if len(stream_examples) > 0:
            first_example = stream_examples[0]
            print(first_example)
            for i, x in enumerate(first_example[0].tolist()[0]):
                values = []
                for row in stream_examples:
                    try:
                        values.append(row[0].tolist()[0][i])
                    except:
                        print(row)
                
                    # values = np.unique(np.array([x[0].tolist()[0][i] for x in stream_examples]))
                values = np.unique(np.array(values))
                if len(values) < 10:
                    # f.write(f"@ATTRIBUTE x{i}  {{{','.join([str(x) for x in values.tolist()])}}}\n")
                    f.write(f"@ATTRIBUTE x{i}  NUMERIC\n")
                else:
                    f.write(f"@ATTRIBUTE x{i}  NUMERIC\n")

            for i, y in enumerate(first_example[1].tolist()):
                values = np.unique(np.array([y[1].tolist()[i] for y in stream_examples]))
                if len(values) < 10:
                    # f.write(f"@ATTRIBUTE y{i}  {{{','.join([str(x) for x in values.tolist()])}}}\n")
                    f.write(f"@ATTRIBUTE y{i}  NUMERIC\n")
                else:
                    f.write(f"@ATTRIBUTE y{i}  NUMERIC\n")
            f.write(f"@DATA\n")
            for l in stream_examples:
                for x in l[0].tolist()[0]:
                    f.write(f"{x},")
                for y in l[1].tolist():
                    f.write(f"{y},")
                f.write(f"\n")

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--directory",
        help="tdata generator for stream", default="datastreams")
ap.add_argument("-hd", "--header", type=int, default = None,
    help="read header")
args = vars(ap.parse_args())

datastream_filename = args['directory']
read_header = args['header']
df = pd.read_csv(datastream_filename, header= read_header, low_memory=False)
df = df.dropna()

for c_i,c in enumerate(df.columns):
    if pd.api.types.is_string_dtype(df[c]):
        # df[c] = pd.to_numeric(df[c], errors = 'ignore')
        try:
            df[c] = df[c].astype(float)
        except Exception as e:
            print(e)
            print(df[c])
            print(f"Factoizing {c}")
            print(pd.factorize(df[c])[0].shape)
            df[c] = pd.factorize(df[c])[0]

stream_examples = []
for row in df.iterrows():
    row_v = row[1].values
    X = np.asarray([row_v[:-1]])
    y = np.asarray([row_v[-1]])
    # print(f"{X} : {y}")
    stream_examples.append((X,y))

saveStreamToArff(f"{datastream_filename[:-4]}.ARFF", stream_examples)

