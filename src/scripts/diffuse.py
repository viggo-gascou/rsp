from rsp.loading import load_model

sd = load_model("pixel", device="cuda", h_space="after", num_inference_steps=10)
# h_space Is the sementic latent space defined as ["before", middle","after"] the
# middle convolution in the U-net
q_original = sd.sample(seed=76)

print(q_original.hs.shape)
