import pymongo


class Database:
    def __init__(self):
        self.myclient = pymongo.MongoClient("Your Serever")
        self.mydb = self.myclient["OSS"]
        self.mycol = self.mydb["data"]

    def insert_one(self, input, output):
        try:
            input_data = {"input": input}
            count = self.mycol.count_documents(input_data)
            input_data["count"] = count + 1
            input_data["output"] = output
            self.mycol.insert_one(input_data)
            return "The data has been inserted into the database successfully."
        except:
            return "Insert failed."

    def find_all(self):
        data_set = []
        try:
            for data in self.mycol.find():
                # print(data)
                data_set.append([data["input"] + "-" + str(data["count"]), data["output"]])
            return data_set
        except:
            return "Query failed."

    def find_one(self, input):
        data = self.mycol.find_one({"input": input})
        return {data["input"], data["output"]}

    def find_many(self, input):
        data_set = []

        for data in self.mycol.find({"input": input}):
            data_set.append([data["input"] + "-" + str(data["count"]), data["output"]])
        return data_set

    def delete_one(self, input, times):
        try:
            self.mycol.delete_one({"input": input, "count": times})
            return "Delete successfully."
        except:
            return "Delete failed."
    def delete_many(self, input):
        try:
            self.mycol.delete_many({"input": input})
            return "Delete successfully."
        except:
            return "Delete failed."


if __name__ == '__main__':
    database = Database()
    # print(database.insert_one("45-10-6-6-4", "(3, 4, 7, 19, 40, 45), (3, 19, 20, 21, 41, 43), (3, 4, 7, 20, 40, 43)"))
    # print(database.delete_one("45-10-6-6-4",1))
    # print(database.find_many("45-10-6-6-4"))
    print(database.find_one("45-16-6-5-4"))
