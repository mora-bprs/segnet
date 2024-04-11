root_path = "content"
lr = 5e-4
batch_size = 10
print_every = 5
eval_every = 5
bc_train = 367 // batch_size  # mini_batch train
bc_eval = 101 // batch_size  # mini_batch validation
epochs = 100
