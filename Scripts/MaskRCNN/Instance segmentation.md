# Instance Segmentation Methods

1. **Segment Proposals Approach**:
   - Many instance segmentation methods are based on segment proposals, leveraging the effectiveness of R-CNN.
   - Earlier methods relied on bottom-up segments for generating segment proposals.

2. **DeepMask and Fast R-CNN**:
   - DeepMask and subsequent works learn to propose segment candidates, which are then classified by Fast R-CNN.
   - These methods prioritize segmentation before recognition, leading to slower and less accurate performance.

3. **Complex Multiple-Stage Cascade**:
   - Dai et al. proposed a complex multiple-stage cascade that predicts segment proposals from bounding-box proposals, followed by classification.
   - This approach involves multiple stages and can be computationally intensive.

4. **Parallel Prediction of Masks and Class Labels**:
   - The method described in the paragraph is based on parallel prediction of masks and class labels, which is simpler and more flexible compared to the multi-stage cascade approach.

5. **Fully Convolutional Instance Segmentation (FCIS)**:
   - Li et al. combined segment proposal and object detection systems for "fully convolutional instance segmentation" (FCIS).
   - FCIS predicts position-sensitive output channels fully convolutionally, addressing object classes, boxes, and masks simultaneously, leading to faster processing.
   - However, FCIS exhibits systematic errors on overlapping instances and creates spurious edges, indicating challenges in segmenting instances accurately.
