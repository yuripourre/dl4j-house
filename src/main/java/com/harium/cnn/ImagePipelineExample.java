package com.harium.cnn;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.FlipImageTransform;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.ResizeImageTransform;
import org.datavec.image.transform.WarpImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Created by susaneraly on 6/9/16.
 */

// Forked from:
// https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/dataexamples/ImagePipelineExample.java

// Networks from:
// https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/dataexamples/MnistImagePipelineExample.java
// https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/convolution/AnimalsClassification.java
public class ImagePipelineExample {

    private static final NeuralNetwork NEURAL_NETWORK = NeuralNetwork.CUSTOM;
    //private static final NeuralNetwork NEURAL_NETWORK = NeuralNetwork.ALEXNET;
    //private static final NeuralNetwork NEURAL_NETWORK = NeuralNetwork.LENET;

    private static final Train TRAIN = Train.FAST;
    //private static final Train TRAIN = Train.FULL;

    private static final boolean SAVE = false;

    private static final int batchSize = 10; // Minibatch size. Here: The number of images to fetch for each call to dataIter.next().
    private static final int iterations = 1;
    private  static int epochs = 50;

    //Images are of format given by allowedExtension
    private static final String[] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;

    private static final long seed = 12345;
    private static final Random randNumGen = new Random(seed);

    private static final int height = 100;
    private static final int width = 100;
    private static final int channels = 3;
    static int outputNum; // Number of labels, updated automatically

    //public final static String DATABASE_NAME = "house-db"; // Full database
    public final static String DATABASE_NAME = "small-db"; // Small database

    private static final File BASE_DIR = new File(System.getProperty("user.dir"));
    final static String IMAGES_DIR = BASE_DIR + File.separator + DATABASE_NAME;

    public static void main(String[] args) throws Exception {
        //DIRECTORY STRUCTURE:
        //Images in the dataset have to be organized in directories by class/label.
        //In this example there are ten images in three classes
        //Here is the directory structure
        //                                    parentDir
        //                                  /    |     \
        //                                 /     |      \
        //                            labelA  labelB   labelC
        //
        //Set your data up like this so that labels from each label/class live in their own directory
        //And these label/class directories live together in the parent directory
        //
        //
        File parentDir = new File(IMAGES_DIR);
        //Files in directories under the parent dir that have "allowed extensions" split needs a random number generator for reproducibility when splitting the files into TRAIN and test
        FileSplit filesInDir = new FileSplit(parentDir, allowedExtensions, randNumGen);

        //You do not have to manually specify labels. This class (instantiated as below) will
        //parse the parent dir and use the name of the subdirectories as label/class names
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        //The balanced path filter gives you fine tune control of the min/max cases to load for each class
        //Below is a bare bones version. Refer to javadoc for details
        BalancedPathFilter pathFilter = new BalancedPathFilter(randNumGen, allowedExtensions, labelMaker);

        //Split the image files into TRAIN and test. Specify the TRAIN test split as 80%,20%
        InputSplit[] filesInDirSplit = filesInDir.sample(pathFilter, 80, 20);
        InputSplit trainData = filesInDirSplit[0];
        InputSplit testData = filesInDirSplit[1];

        //Specifying a new record reader with the height and width you want the images to be resized to.
        //Note that the images in this example are all of different size
        //They will all be resized to the height and width specified below
        ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);

        //Initialize the record reader with the TRAIN data
        recordReader.initialize(trainData, null);
        outputNum = recordReader.numLabels();

        //int labelIndex = 1; // Index of the label Writable (usually an IntWritable), as obtained by recordReader.next()
        // List<Writable> lw = recordReader.next();
        // then lw[0] =  NDArray shaped [1,3,50,50] (1, height, width, channels)
        //      lw[0] =  label as integer.

        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);

        MultiLayerNetwork network = buildNeuralNetwork();

        trainModel(trainData, recordReader, scaler, network);

        System.out.println("Evaluating Model");
        Evaluation eval = evaluateModel(testData, recordReader, scaler, network);

        System.out.println(eval.stats(true));

        // Example on how to get predict results with trained model. Result for first example in minibatch is printed
        recordReader.initialize(testData, null);
        DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, outputNum);
        DataSet testDataSet = dataIter.next();

        List<String> allClassLabels = recordReader.getLabels();
        int labelIndex = testDataSet.getLabels().argMax(1).getInt(0);
        int[] predictedClasses = network.predict(testDataSet.getFeatures());
        String expectedResult = allClassLabels.get(labelIndex);
        String modelPrediction = allClassLabels.get(predictedClasses[0]);
        System.out.print("\nFor a single example that is labeled " + expectedResult + " the model predicted " + modelPrediction + "\n\n");

        if (SAVE) {
            System.out.println("Save model....");
            String basePath = System.getProperty("user.dir") + "src/main/resources/";
            ModelSerializer.writeModel(network, basePath + "model.bin", true);
        }
    }

    private static void trainModel(InputSplit trainData, ImageRecordReader recordReader, DataNormalization scaler, MultiLayerNetwork network) throws IOException {
        switch (TRAIN) {
            case FAST:
                fastTrainModel(trainData, recordReader, scaler, network);
                break;
            case FULL:
                fullTrainModel(trainData, recordReader, scaler, network);
                break;
        }
    }

    private static void fastTrainModel(InputSplit trainData, ImageRecordReader recordReader, DataNormalization scaler, MultiLayerNetwork network) throws IOException {
        ImageTransform transform = new ResizeImageTransform(width, height);

        System.out.print("\nTraining on transformation: " + transform.getClass().toString() + "\n\n");
        recordReader.initialize(trainData, transform);
        DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, outputNum);
        scaler.fit(dataIter);
        dataIter.setPreProcessor(scaler);
        MultipleEpochsIterator trainIter = new MultipleEpochsIterator(epochs, dataIter);
        network.fit(trainIter);

    }

    private static void fullTrainModel(InputSplit trainData, ImageRecordReader recordReader, DataNormalization scaler, MultiLayerNetwork network) throws IOException {
        ImageTransform flipTransform1 = new FlipImageTransform(randNumGen);
        ImageTransform flipTransform2 = new FlipImageTransform(new Random(seed * 3));
        ImageTransform warpTransform = new WarpImageTransform(randNumGen, 42);
        //ImageTransform colorTransform = new ColorConversionTransform(new Random(seed), COLOR_BGR2YCrCb);
        List<ImageTransform> transforms = Arrays.asList(new ImageTransform[]{flipTransform1, warpTransform, flipTransform2});

        DataSetIterator dataIter;
        // Train with transformations
        for (ImageTransform transform : transforms) {
            System.out.print("\nTraining on transformation: " + transform.getClass().toString() + "\n\n");
            recordReader.initialize(trainData, transform);
            dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, outputNum);
            scaler.fit(dataIter);
            dataIter.setPreProcessor(scaler);
            MultipleEpochsIterator trainIter = new MultipleEpochsIterator(epochs, dataIter);
            network.fit(trainIter);
        }
    }

    private static Evaluation evaluateModel(InputSplit testData, ImageRecordReader recordReader, DataNormalization scaler, MultiLayerNetwork network) throws IOException {
        recordReader.reset();

        // The model trained on the training dataset split
        // now that it has trained we evaluate against the
        // test data of images the network has not seen

        recordReader.initialize(testData, null);
        DataSetIterator testIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, outputNum);
        scaler.fit(testIter);
        testIter.setPreProcessor(scaler);
        Evaluation eval = network.evaluate(testIter);

        // Evaluate the network
        while (testIter.hasNext()) {
            DataSet next = testIter.next();
            INDArray output = network.output(next.getFeatureMatrix());
            // Compare the Feature Matrix from the model
            // with the labels from the RecordReader
            eval.eval(next.getLabels(), output);
        }
        return eval;
    }

    private static MultiLayerNetwork buildNeuralNetwork() {
        switch (NEURAL_NETWORK) {
            case ALEXNET:
                return alexnetModel();
            case LENET:
                return lenetModel();
            case CUSTOM:
            default:
                return customModel();
        }
    }

    private static MultiLayerNetwork customModel() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT)
                .iterations(1)
                .learningRate(0.006)
                .updater(Updater.NESTEROVS)
                .regularization(true).l2(1e-4)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(height * width * channels)
                        .nOut(100)
                        .activation(Activation.RELU)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(100)
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .pretrain(false).backprop(true)
                .setInputType(InputType.convolutional(height, width, channels))
                .build();

        return new MultiLayerNetwork(conf);
    }

    private static MultiLayerNetwork lenetModel() {
        /**
         * Revised Lenet Model approach developed by ramgo2 achieves slightly above random
         * Reference: https://gist.github.com/ramgo2/833f12e92359a2da9e5c2fb6333351c5
         **/
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .regularization(false).l2(0.005) // tried 0.0001, 0.0005
                .activation(Activation.RELU)
                .learningRate(0.0001) // tried 0.00001, 0.00005, 0.000001
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Nesterovs(0.9))
                .list()
                .layer(0, convInit("cnn1", channels, 50, new int[]{5, 5}, new int[]{1, 1}, new int[]{0, 0}, 0))
                .layer(1, maxPool("maxpool1", new int[]{2, 2}))
                .layer(2, conv5x5("cnn2", 100, new int[]{5, 5}, new int[]{1, 1}, 0))
                .layer(3, maxPool("maxool2", new int[]{2, 2}))
                .layer(4, new DenseLayer.Builder().nOut(500).build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .build())
                .backprop(true).pretrain(false)
                .setInputType(InputType.convolutional(height, width, channels))
                .build();

        return new MultiLayerNetwork(conf);
    }

    private static MultiLayerNetwork alexnetModel() {
        /**
         * AlexNet model interpretation based on the original paper ImageNet Classification with Deep Convolutional Neural Networks
         * and the imagenetExample code referenced.
         * http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
         **/

        double nonZeroBias = 1;
        double dropOut = 0.5;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.DISTRIBUTION)
                .dist(new NormalDistribution(0.0, 0.01))
                .activation(Activation.RELU)
                .updater(new Nesterovs(0.9))
                .iterations(iterations)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer) // normalize to prevent vanishing or exploding gradients
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(1e-2)
                .biasLearningRate(1e-2 * 2)
                .learningRateDecayPolicy(LearningRatePolicy.Step)
                .lrPolicyDecayRate(0.1)
                .lrPolicySteps(100000)
                .regularization(true)
                .l2(5 * 1e-4)
                .list()
                .layer(0, convInit("cnn1", channels, width / 2, new int[]{11, 11}, new int[]{4, 4}, new int[]{3, 3}, 0))
                .layer(1, new LocalResponseNormalization.Builder().name("lrn1").build())
                .layer(2, maxPool("maxpool1", new int[]{3, 3}))
                .layer(3, conv5x5("cnn2", 256, new int[]{1, 1}, new int[]{2, 2}, nonZeroBias))
                .layer(4, new LocalResponseNormalization.Builder().name("lrn2").build())
                .layer(5, maxPool("maxpool2", new int[]{3, 3}))
                .layer(6, conv3x3("cnn3", 384, 0))
                .layer(7, conv3x3("cnn4", 384, nonZeroBias))
                .layer(8, conv3x3("cnn5", 256, nonZeroBias))
                .layer(9, maxPool("maxpool3", new int[]{3, 3}))
                .layer(10, fullyConnected("ffn1", 4096, nonZeroBias, dropOut, new GaussianDistribution(0, 0.005)))
                .layer(11, fullyConnected("ffn2", 4096, nonZeroBias, dropOut, new GaussianDistribution(0, 0.005)))
                .layer(12, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .name("output")
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .build())
                .backprop(true)
                .pretrain(false)
                .setInputType(InputType.convolutional(height, width, channels))
                .build();

        return new MultiLayerNetwork(conf);
    }

    // Helper Methods
    private static ConvolutionLayer convInit(String name, int in, int out, int[] kernel, int[] stride, int[] pad, double bias) {
        return new ConvolutionLayer.Builder(kernel, stride, pad).name(name).nIn(in).nOut(out).biasInit(bias).build();
    }

    private static ConvolutionLayer conv3x3(String name, int out, double bias) {
        return new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1}).name(name).nOut(out).biasInit(bias).build();
    }

    private static ConvolutionLayer conv5x5(String name, int out, int[] stride, int[] pad, double bias) {
        return new ConvolutionLayer.Builder(new int[]{5, 5}, stride, pad).name(name).nOut(out).biasInit(bias).build();
    }

    private static SubsamplingLayer maxPool(String name, int[] kernel) {
        return new SubsamplingLayer.Builder(kernel, new int[]{2, 2}).name(name).build();
    }

    private static DenseLayer fullyConnected(String name, int out, double bias, double dropOut, Distribution dist) {
        return new DenseLayer.Builder().name(name).nOut(out).biasInit(bias).dropOut(dropOut).dist(dist).build();
    }
}