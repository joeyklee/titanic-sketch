class NeuralNet {
    constructor(options) {
        this.config = {
            inputs: options.inputs || 3,
            outputs: options.outputs || 2,
            debug: options.debug || true,
            outputLabels: options.outputLabels || [],
            inputLabels: options.inputLabels || [],
            batchSize:options.batchSize || 64,
            epochs: options.epochs || 32,
            NO_VAL: null
        }
        this.config.NO_VAL = this.config.outputs;
        this.model = null;
    }


    async createModel() {

        this.model = tf.sequential();

        const hidden = tf.layers.dense({
            units: 16,
            inputShape: [this.config.inputs],
            activation: 'relu'
        });
        
        const output = tf.layers.dense({
            units: this.config.outputs,
            activation: 'softmax'
        });
        this.model.add(hidden);
        // this.model.add(tf.layers.dense({units: 10, activation: 'relu'}));
        this.model.add(output);

        this.model.compile({
            optimizer: tf.train.adam(),
            // This loss breaks things?
            // loss: 'sparseCategoricalCrossentropy',
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy'],
        });

        if (this.config.debug) {
            tfvis.show.modelSummary({
                name: 'Model Summary'
            }, this.model);
        }
    }

    async loadData(DATA_URL, dataType) {
        const outputLabel = this.config.outputLabels[0];
        const inputLabels = this.config.inputLabels;
        console.log(inputLabels)

        switch (dataType) {
            case 'csv':
                let data = await tf.data.csv(DATA_URL, {
                    columnConfigs: {
                        [outputLabel]: {
                            isLabel: true
                        }
                    }
                });
                data = await data.toArray();
                
                if (this.config.debug) {
                    const values = inputLabels.map( label => {
                        return data.map( item => {
                            return {
                                x: item.xs[label],
                                y: item.ys[outputLabel]
                            }
                        })
                    })

                    tfvis.render.scatterplot({
                        name: 'debug mode'
                    }, {
                        values
                    }, {
                        xLabel: 'X',
                        yLabel: 'Y',
                        height: 300
                    });
                }
                return data;
            default:
                console.log('error!')
                return;
        }
    }


    async parseData(data) {
        return tf.tidy(() => {
            const outputLabel = this.config.outputLabels[0];
            const inputLabels = this.config.inputLabels;
            // Step 1. Shuffle the data    
            tf.util.shuffle(data);

            // Step 2. Convert data to Tensor
            // const inputs = data.map(d => inputLabels.map(header => d.xs[header]));
            const inputs = inputLabels.map(header => data.map(d => d.xs[header]))
            const labels = data.map(d => d.ys[outputLabel]);
            
            const inputTensor = tf.tensor(inputs); 
            const labelTensor = tf.oneHot(tf.tensor1d(labels, 'int32'), this.config.NO_VAL );

            //Step 3. Normalize the data to the range 0 - 1 using min-max scaling
            const inputMax = inputTensor.max();
            const inputMin = inputTensor.min();  
            const labelMax = labelTensor.max();
            const labelMin = labelTensor.min();

            const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin)).flatten().reshape([ data.length, this.config.inputs]);

            console.log()
            const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));
        
            // inputTensor.max(1).print();
            return {
                inputs: normalizedInputs, // normalizedInputs,
                labels: normalizedLabels,
                // Return the min/max bounds so we can use them later.
                inputMax,
                inputMin,
                labelMax,
                labelMin,
              }


        })

    }

    async train(inputs, labels) {
        const batchSize = this.config.batchSize;
        const epochs = this.config.epochs;
        // console.log(inputs, labels)
        
        await this.model.fit(inputs, labels, {
            shuffle: true,
            batchSize: batchSize,
            epochs: epochs,
            callbacks:{
                onEpochEnd: (epoch, logs) => {
                  console.log(`Epoch: ${epoch} - accuracy: ${logs.loss.toFixed(3)}`);
                }
              },
          });
       
    }

    async  predict(sample) {
        console.log(sample)
        let result = this.model.predict(tf.tensor(sample, [1,sample.length])).arraySync();
        
        let maxValue = 0;
        let output = this.config.NO_VAL;

        for (var i = 0; i < this.config.inputs; i++) {
          if (result[0][i] > maxValue) {
            output = i;
          }
        }

        return output;
      
      }
}