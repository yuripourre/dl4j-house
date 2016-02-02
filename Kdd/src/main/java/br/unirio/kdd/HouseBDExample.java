package br.unirio.kdd;

import br.unirio.kdd.datasets.iterator.impl.ImageDataSetIterator;
import org.apache.commons.io.FileUtils;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.LFWDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * @author Adam Gibson
 */
public class HouseBDExample {
    private static final Logger log = LoggerFactory.getLogger(HouseBDExample.class);

    static final int numRows = 64;
    static final int numColumns = 64;
    static int nChannels = 3;//Number of image channels (RGB = 3)
    static int outputNum = 4;//Number of tags
    static int numSamples = 400;
    static int batchSize = 330;//Lower than numSamples
    static int iterations = 12;
    static int seed = 123;
    static int splitTrainNum = (int) (batchSize * .8);
    static int listenerFreq = iterations / 5;

    static SplitTestAndTrain trainTest;
    static DataSet trainInput;
    static List<INDArray> testInput = new ArrayList<>();
    static List<INDArray> testLabels = new ArrayList<>();

    private static String FILE_PARAMS = "housebd_params.bin";
    private static String FILE_MODEL = "housebd_model.json";

    public static void main(String[] args) throws IOException {

        log.info("Load data....");
        DataSetIterator iterator = new ImageDataSetIterator(batchSize, numSamples, numRows, numColumns);

        outputNum = iterator.totalOutcomes();

        log.info("Build model....");
        //MultiLayerNetwork model = loadTrainedModel(iterator);
        MultiLayerNetwork model = buildModel();
        trainModel(iterator, model);
        saveModel(model);

        log.info("Evaluate model....");
        Evaluation eval = new Evaluation(outputNum);
        for (int i = 0; i < testInput.size(); i++) {
            INDArray output = model.output(testInput.get(i));
            eval.eval(testLabels.get(i), output);
        }
        log.info("****************Final Eval********************");
        INDArray output = model.output(testInput.get(0));
        eval.eval(testLabels.get(0), output);
        log.info(eval.stats());
        log.info("****************Example finished********************");
    }

    private static void trainModel(DataSetIterator iterator, MultiLayerNetwork model) {
        DataSet lfwNext;
        log.info("Train model....");
        model.setListeners(Arrays.asList((IterationListener) new ScoreIterationListener(listenerFreq)));

        while (iterator.hasNext()) {
            lfwNext = iterator.next();
            lfwNext.scale();
            trainTest = lfwNext.splitTestAndTrain(splitTrainNum, new Random(seed)); // train set that is the result
            trainInput = trainTest.getTrain(); // get feature matrix and labels for training
            testInput.add(trainTest.getTest().getFeatureMatrix());
            testLabels.add(trainTest.getTest().getLabels());
            model.fit(trainInput);
        }
    }

    private static MultiLayerNetwork buildModel() {
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .activation("relu")
                .weightInit(WeightInit.XAVIER)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(0.01)
                .momentum(0.9)
                .regularization(true)
                .updater(Updater.ADAGRAD)
                .useDropConnect(true)
                .list(6)
                .layer(0, new ConvolutionLayer.Builder(4, 4)
                        .name("cnn1")
                        .nIn(nChannels)
                        .stride(1, 1)
                        .nOut(20)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2})
                        .name("pool1")
                        .build())
                .layer(2, new ConvolutionLayer.Builder(3, 3)
                        .name("cnn2")
                        .stride(1,1)
                        .nOut(40)
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2})
                        .name("pool2")
                        .build())
                .layer(2, new ConvolutionLayer.Builder(3, 3)
                        .name("cnn3")
                        .stride(1,1)
                        .nOut(60)
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2})
                        .name("pool3")
                        .build())
                .layer(2, new ConvolutionLayer.Builder(2, 2)
                        .name("cnn3")
                        .stride(1,1)
                        .nOut(80)
                        .build())
                .layer(4, new DenseLayer.Builder()
                        .name("ffn1")
                        .nOut(160)
                        .dropOut(0.5)
                        .build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .activation("softmax")
                        .build())
                .backprop(true).pretrain(false);
        new ConvolutionLayerSetup(builder,numRows,numColumns,nChannels);

        MultiLayerConfiguration conf = builder.build();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        return model;
    }

    private static void saveModel(MultiLayerNetwork model) throws IOException {
        OutputStream fos = Files.newOutputStream(Paths.get(FILE_PARAMS));
        DataOutputStream dos = new DataOutputStream(fos);
        Nd4j.write(model.params(), dos);
        dos.flush();
        dos.close();
        FileUtils.write(new File(FILE_MODEL), model.getLayerWiseConfigurations().toJson());
    }

    private static MultiLayerNetwork loadTrainedModel(DataSetIterator iterator) {
        MultiLayerNetwork model;
        try {
            model = loadModel();
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }

        DataSet lfwNext;
        log.info("Train model....");
        model.setListeners(Arrays.asList((IterationListener) new ScoreIterationListener(listenerFreq)));

        while (iterator.hasNext()) {
            lfwNext = iterator.next();
            lfwNext.scale();
            //trainTest = lfwNext.splitTestAndTrain(splitTrainNum, new Random(seed)); // train set that is the result
            //trainInput = lfwNext.get.getTrain(); // get feature matrix and labels for training
            testInput.add(lfwNext.getFeatures());
            testLabels.add(lfwNext.getLabels());
        }

        //trainModel(iterator, model);

        return model;
    }

    private static MultiLayerNetwork loadModel() throws IOException {
        MultiLayerConfiguration confFromJson = MultiLayerConfiguration.fromJson(FileUtils.readFileToString(new File(FILE_MODEL)));
        DataInputStream dis = new DataInputStream(new FileInputStream(FILE_PARAMS));
        INDArray newParams = Nd4j.read(dis);
        dis.close();
        MultiLayerNetwork savedNetwork = new MultiLayerNetwork(confFromJson);
        savedNetwork.init();
        savedNetwork.setParameters(newParams);

        return savedNetwork;
    }

}

