from message_classifier import read_tensor_flow_dataset

message_train, message_test, label_train, label_test = read_tensor_flow_dataset('tf_vuln_dataset.csv')

pos = 0
neg = 0

for label in label_train:
    if label == 0:
        neg += 1
    else:
        pos += 1

for label in label_test:
    if label == 0:
        neg += 1
    else:
        pos += 1

print(pos)
print(neg)