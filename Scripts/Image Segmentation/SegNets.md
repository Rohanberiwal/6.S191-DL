# SegNets

SegNets are fully convolutional encoder-decoder architectures designed for image segmentation.
## Architecture

- **SegNets**: These are composed of an encoder and a decoder.
  - **Encoder**: Typically uses the VGG-16 network (a pre-trained convolutional network). The fully connected layers are not included.
  - **Decoder**: Maps the low-resolution encoder feature maps back to the input resolution.

## Differences Between SegNets and FCNs

- **Similarity**: Both SegNets and FCNs use an encoder-decoder architecture.
- **Major Difference**: The primary difference lies in the decoding phase:
  - **FCNs**: Have a single up-sampling layer at the end of the computation.
  - **SegNets**: Feature multiple decoding units, each specifically trained to perform up-sampling.

## Encoder-Decoder Structure

- **Mirror Architecture**: The encoder and decoder units in SegNets are mirrors of each other, meaning their size and dimensions correspond to one another.
- **Computation Difference**:
  - **No Transpose/Dilated/Atrous Convolutions**: SegNets do not use these for up-sampling.
  - **Pooling Indices**: SegNets use pooling indices obtained during the encoding phase for up-sampling in the decoder phase.

## Max Pool Indices

- **Definition**: Locations of the maximum feature value in each pooling window for each encoder feature map.
- **Efficiency**: Storing max pool indices is more memory-efficient.

## Decoding Process

- **Sparse Feature Maps**: The decoder uses the max pool indices to produce sparse up-sampled feature maps.
- **Repeated Process**: This up-sampling process is repeated to obtain the final up-sampled image.

## Loss Function

- **Cross-Entropy Loss**: The loss function for SegNets is the sum of pixel-wise cross-entropy losses.

## Output

- **Output Volume**: Both FCNs and SegNets produce an output volume where the number of output channels is equal to the number of input channels plus one (for the background).

### Example Diagram

Input Image -> Encoder (VGG-16 without FC layers) -> Low-Res Feature Maps
Low-Res Feature Maps -> Decoder (using pooling indices) -> Output Image
