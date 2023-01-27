import tensorflow_datasets.public_api as tfds
import tensorflow as tf
GENE_NUMBER = 12021

class MyDataset(tfds.core.GeneratorBasedBuilder):
  """_summary_

  Args:
    tfds (_type_): _description_

  Yields:
    _type_: _description_
  """
    
  VERSION = tfds.core.Version('0.1.0')

  def _info(self):
    return tfds.core.DatasetInfo(
        builder=self,
        # 这是将在数据集页面上显示的描述。
        description=("This is the dataset for GeneNet. It contains different"),
        # tfds.features.FeatureConnectors
        features=tfds.features.FeaturesDict({
            "tensor_description": tfds.features.Text(),
            "tensor": tfds.features.Tensor(shape=(GENE_NUMBER, 1), dtype=tf.float64),
            "label": tfds.features.ClassLabel(num_classes=2),
        }),
        # 如果特征中有一个通用的（输入，目标）元组，
        # 请在此处指定它们。它们将会在
        # builder.as_dataset 中的 
        # as_supervised=True 时被使用。
        supervised_keys=("tensor", "label"),
    )

  def _split_generators(self, dl_manager):
    # 下载数据并定义划分
    # dl_manager 是一个 tfds.download.DownloadManager，其能够被用于
    # 下载并提取 URLs
    dl_paths = dl_manager.download_and_extract({
      'foo': 'https://static-content.springer.com/esm/art%3A10.1038%2Fs41588-018-0078-z/MediaObjects/41588_2018_78_MOESM6_ESM.xlsx',
      'bar': 'https://example.com/bar.zip',
  })
    
    pass  # TODO

  def _generate_examples(self):
    # 从数据集中产生样本
    yield 'key', {}