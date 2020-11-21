# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter
from scrapy.exporters import CsvItemExporter, JsonItemExporter


class AtlasisscrapyPipeline:
    def open_spider(self, spider):
        self.json_file = open("ISExport.json", 'w+b')
        # self.csv_file = open("ISExport.csv", 'w+b')

        self.json_exporter = JsonItemExporter(self.json_file)
        # self.csv_exporter = CsvItemExporter(self.csv_file)

        self.json_exporter.start_exporting()
        # self.csv_exporter.start_exporting()

    def close_spider(self, spider):
        self.json_exporter.finish_exporting()
        # self.csv_exporter.finish_exporting()

        self.json_file.close()
        # self.csv_file.close()


    def process_item(self, item, spider):
        self.json_exporter.export_item(item)
        # self.csv_exporter.export_item(item)
        return item
