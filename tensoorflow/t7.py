import tensorflow as tf
flags=tf.app.flags
logging=tf.logging
flags.DEFINE_string("p1","default","desc")
flags.DEFINE_bool("p2","p2","p2")
FLAGS=flags.FLAGS
def main(_):
    print(FLAGS.p1)
if __name__ == '__main__':
    tf.app.run()