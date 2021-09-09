import tflite_runtime.interpreter as tflite

#tf.compat.v1.enable_eager_execution()
#tf.enable_eager_execution()

interpreter = tflite.Interpreter(model_path=args.model_file)

#hello = tf.constant('Hello, TensorFlow!')

print(hello)

