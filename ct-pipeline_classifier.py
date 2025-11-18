"""
Standalone Python Script for Local Execution (2 Labels)
Run this script to process documents without Cloud Functions
"""

import json
import re
import time
from typing import List
from google.cloud import documentai_v1beta3 as documentai
from google.cloud import storage
from google.api_core.client_options import ClientOptions


# ===== CONFIGURATION =====
PROJECT_ID = "YOUR_PROJECT_ID"
LOCATION = "us"
PROCESSOR_ID = "YOUR_PROCESSOR_ID"
BUCKET_NAME = "continuoustraining"

# 2 Labels
ENTITY_TYPE_MAP = {
    "AP_invoice": "AP_invoice",
    "Tax_Certificates": "Tax_Certificates"
}


class ContinuousTrainingPipeline:
    def __init__(self):
        self.project_id = PROJECT_ID
        self.location = LOCATION
        self.processor_id = PROCESSOR_ID
        self.bucket_name = BUCKET_NAME
        
        self.client_options = ClientOptions(
            api_endpoint=f"{LOCATION}-documentai.googleapis.com"
        )
        self.processor_client = documentai.DocumentProcessorServiceClient(
            client_options=self.client_options
        )
        self.document_service_client = documentai.DocumentServiceClient(
            client_options=self.client_options
        )
        self.storage_client = storage.Client(project=PROJECT_ID)
        
        self.processor_name = f"projects/{PROJECT_ID}/locations/{LOCATION}/processors/{PROCESSOR_ID}"
        self.dataset_name = f"{self.processor_name}/dataset"

    @staticmethod
    def normalize_gcs_uri(uri: str) -> str:
        uri = uri.strip()
        if not uri.startswith("gs://"):
            raise ValueError("URI must start with gs://")
        body = re.sub(r"/+", "/", uri[5:])
        return "gs://" + (body if body.endswith("/") else body + "/")

    def create_or_update_dataset(self):
        print("\n=== Creating/Updating Dataset ===")
        dataset = documentai.Dataset(
            name=self.dataset_name,
            gcs_managed_config=documentai.Dataset.GCSManagedConfig(
                gcs_prefix=documentai.GcsPrefix(
                    gcs_uri_prefix=f"gs://{self.bucket_name}/dataset/"
                )
            )
        )
        try:
            self.document_service_client.update_dataset(
                request=documentai.UpdateDatasetRequest(dataset=dataset)
            )
            print("✓ Dataset created/updated")
        except Exception as e:
            if "ALREADY_EXISTS" in str(e) or "DATASET_INITIALIZED" in str(e):
                print("✓ Dataset already exists")
            else:
                raise

    def get_folder_structure(self):
        print("\n=== Scanning Bucket ===")
        bucket = self.storage_client.bucket(self.bucket_name)
        blobs = bucket.list_blobs(prefix="raw_documents/")
        
        folder_structure = {}
        for blob in blobs:
            if not blob.name.lower().endswith(".pdf"):
                continue
            parts = blob.name.split("/")
            if len(parts) > 1:
                doc_type = parts[1]
                if doc_type in ENTITY_TYPE_MAP:
                    folder_structure.setdefault(doc_type, []).append(
                        f"gs://{self.bucket_name}/{blob.name}"
                    )
        
        for k, v in folder_structure.items():
            print(f" - {k}: {len(v)} PDFs")
        
        return folder_structure

    def batch_process_prefix(self, input_prefix: str, output_gcs_uri: str):
        print(f"\n=== Batch Processing ===")
        print(f"Input: {input_prefix}")
        print(f"Output: {output_gcs_uri}")
        
        output_gcs_uri = self.normalize_gcs_uri(output_gcs_uri)

        gcs_input_prefix = documentai.GcsPrefix(gcs_uri_prefix=input_prefix)
        input_config = documentai.BatchDocumentsInputConfig(gcs_prefix=gcs_input_prefix)
        
        gcs_output_config = documentai.DocumentOutputConfig.GcsOutputConfig(
            gcs_uri=output_gcs_uri
        )
        output_config = documentai.DocumentOutputConfig(
            gcs_output_config=gcs_output_config
        )

        request = documentai.BatchProcessRequest(
            name=self.processor_name,
            input_documents=input_config,
            document_output_config=output_config
        )

        operation = self.processor_client.batch_process_documents(request=request)
        print(f"Operation: {operation.operation.name}")
        
        operation.result(timeout=1800)
        print("✓ Batch processing complete")

    def add_classifier_labels(self, gcs_prefix: str, doc_type: str):
        print(f"\n=== Adding Labels: {doc_type} ===")
        
        entity_type = ENTITY_TYPE_MAP[doc_type]
        
        bucket_name, _, prefix = gcs_prefix[5:].partition("/")
        bucket = self.storage_client.bucket(bucket_name)
        json_blobs = [
            b for b in bucket.list_blobs(prefix=prefix) 
            if b.name.lower().endswith(".json")
        ]
        
        print(f"Found {len(json_blobs)} JSON files")
        
        for blob in json_blobs:
            doc_json = json.loads(blob.download_as_text())
            
            doc_json["entities"] = [{
                "type": entity_type,
                "mentionText": entity_type,
                "confidence": 1.0,
                "pageAnchor": {"pageRefs": [{"page": 0}]}
            }]
            
            blob.upload_from_string(
                json.dumps(doc_json), 
                content_type="application/json"
            )
        
        print(f"✓ Labeled {len(json_blobs)} documents")

    def import_to_dataset(self, gcs_uri_prefix: str, split_ratio: float = 0.8):
        print(f"\n=== Importing to Dataset ===")
        gcs_uri_prefix = self.normalize_gcs_uri(gcs_uri_prefix)
        
        bucket_name, _, prefix = gcs_uri_prefix[5:].partition("/")
        bucket = self.storage_client.bucket(bucket_name)
        
        existing_names = set()
        try:
            for doc_meta in self.document_service_client.list_documents(
                request={"parent": self.dataset_name}
            ):
                if hasattr(doc_meta, "display_name") and doc_meta.display_name:
                    existing_names.add(doc_meta.display_name)
        except:
            pass
        
        print(f"Already in dataset: {len(existing_names)}")
        
        json_blobs = [
            b for b in bucket.list_blobs(prefix=prefix) 
            if b.name.lower().endswith(".json")
        ]
        new_blobs = [
            b for b in json_blobs 
            if b.name.split("/")[-1] not in existing_names
        ]
        
        if not new_blobs:
            print("✓ No new documents to import")
            return
        
        split_idx = max(2, int(len(new_blobs) * split_ratio))
        train_blobs = new_blobs[:split_idx]
        test_blobs = new_blobs[split_idx:]
        
        if len(test_blobs) < 2:
            test_blobs = new_blobs[-2:]
            train_blobs = new_blobs[:-2]
        
        print(f"Importing: {len(train_blobs)} train, {len(test_blobs)} test")
        
        if train_blobs:
            self._import_split(train_blobs, gcs_uri_prefix, "TRAIN", bucket)
        
        if test_blobs:
            self._import_split(test_blobs, gcs_uri_prefix, "TEST", bucket)

    def _import_split(self, blobs: List, base_prefix: str, split_type: str, source_bucket):
        ts = int(time.time() * 1000)
        import_prefix = f"{base_prefix.rstrip('/')}_import_{split_type}_{ts}/"
        
        import_bucket_name, _, import_path = import_prefix[5:].partition("/")
        import_bucket = self.storage_client.bucket(import_bucket_name)
        
        for src_blob in blobs:
            dest_path = (import_path + src_blob.name.split("/")[-1]).lstrip("/")
            source_bucket.copy_blob(src_blob, import_bucket, dest_path)
        
        batch_input = documentai.BatchDocumentsInputConfig(
            gcs_prefix=documentai.GcsPrefix(gcs_uri_prefix=import_prefix)
        )
        
        split_enum = (
            documentai.DatasetSplitType.DATASET_SPLIT_TRAIN 
            if split_type == "TRAIN" 
            else documentai.DatasetSplitType.DATASET_SPLIT_TEST
        )
        
        batch_config = documentai.ImportDocumentsRequest.BatchDocumentsImportConfig(
            batch_input_config=batch_input,
            dataset_split=split_enum
        )
        
        request = documentai.ImportDocumentsRequest(
            dataset=self.dataset_name,
            batch_documents_import_configs=[batch_config]
        )
        
        operation = self.document_service_client.import_documents(request=request)
        op_name = operation.operation.name
        print(f"Import {split_type} operation: {op_name}")
        
        ops_client = self.document_service_client.transport.operations_client
        deadline = time.time() + 1800
        
        while time.time() < deadline:
            op = ops_client.get_operation(op_name)
            if op.done:
                if op.error and op.error.code != 0:
                    raise RuntimeError(f"Import {split_type} failed: {op.error.message}")
                print(f"✓ Import {split_type} complete")
                break
            time.sleep(10)
        else:
            raise TimeoutError(f"Import {split_type} timeout")
        
        for blob in import_bucket.list_blobs(prefix=import_path):
            blob.delete()

    def get_dataset_size(self):
        try:
            train_count = 0
            test_count = 0
            
            for doc_meta in self.document_service_client.list_documents(
                request={"parent": self.dataset_name}
            ):
                if hasattr(doc_meta, 'dataset_type'):
                    if doc_meta.dataset_type == documentai.DatasetSplitType.DATASET_SPLIT_TRAIN:
                        train_count += 1
                    elif doc_meta.dataset_type == documentai.DatasetSplitType.DATASET_SPLIT_TEST:
                        test_count += 1
            
            return train_count, test_count
        except:
            return 0, 0

    def train_processor_version(self, version_name: str):
        print(f"\n=== Training Model ===")
        print(f"Version: {version_name}")
        
        request = documentai.TrainProcessorVersionRequest(
            parent=self.processor_name,
            processor_version=documentai.ProcessorVersion(
                display_name=version_name
            )
        )
        
        operation = self.processor_client.train_processor_version(request=request)
        print(f"Operation: {operation.operation.name}")
        print("⏱ Training takes 6-10 hours")
        
        return operation.operation.name


def main():
    """Main execution"""
    print("=" * 60)
    print("Document AI Continuous Training Pipeline (2 Labels)")
    print("=" * 60)
    
    pipeline = ContinuousTrainingPipeline()
    
    try:
        # Step 1: Create/update dataset
        pipeline.create_or_update_dataset()
        
        # Step 2: Get documents from bucket
        folder_structure = pipeline.get_folder_structure()
        
        if not folder_structure:
            print("\n❌ No PDFs found in raw_documents/")
            return
        
        # Step 3: Process each document type
        for doc_type in folder_structure:
            input_prefix = f"gs://{BUCKET_NAME}/raw_documents/{doc_type}/"
            output_prefix = f"gs://{BUCKET_NAME}/processed_documents/{doc_type}/"
            
            # Batch process
            pipeline.batch_process_prefix(input_prefix, output_prefix)
            
            # Add labels
            pipeline.add_classifier_labels(output_prefix, doc_type)
            
            # Import to dataset
            pipeline.import_to_dataset(output_prefix)
        
        # Step 4: Check dataset size and train
        train_count, test_count = pipeline.get_dataset_size()
        print(f"\n=== Dataset Summary ===")
        print(f"Train: {train_count}")
        print(f"Test: {test_count}")
        
        MIN_TRAIN = 8  # 4 per label × 2 labels
        MIN_TEST = 4   # 2 per label × 2 labels
        
        if train_count >= MIN_TRAIN and test_count >= MIN_TEST:
            version_name = f"continuous-training-{int(time.time())}"
            pipeline.train_processor_version(version_name)
        else:
            print(f"\n⚠ Insufficient data for training")
            print(f"Required: {MIN_TRAIN} train, {MIN_TEST} test")
            print(f"Current: {train_count} train, {test_count} test")
        
        print("\n" + "=" * 60)
        print("✓ Pipeline Complete")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        raise


if __name__ == "__main__":
    main()
