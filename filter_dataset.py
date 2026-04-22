import os

allowed_ids = [
"-m-01g317",
"-m-01bqk0",
"-m-01c648",
"-m-01d40f",
"-m-01dxs",
"-m-01mzpv",
"-m-01s55n",
"-m-01y9k5",
"-m-01ww8y",
"-m-0199g",
"-m-01lynh",
"-m-02522",
"-m-04dr76w",
"-m-03bt1vf"
]

label_path = "dataset/train/labels"

for file in os.listdir(label_path):
    file_path = os.path.join(label_path, file)

    with open(file_path, 'r') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        cls = line.split()[0]
        if cls in allowed_ids:
            new_lines.append(line)

    with open(file_path, 'w') as f:
        f.writelines(new_lines)