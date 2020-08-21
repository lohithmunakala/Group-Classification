from bing_image_downloader import downloader

# query_list = ["group of babaies", "group of teenagers", "group of adults"]
# query_list = ["group of babaies",  "group of adults"]
query_list = ["test"]

def imagedownloader(img):

    downloader.download( img,
                         limit = 5,
                         output_dir = "downloads",
                         adult_filter_off = True,
                         force_replace = True )

if __name__ == "__main__":
    for query in query_list:
        imagedownloader(query)
