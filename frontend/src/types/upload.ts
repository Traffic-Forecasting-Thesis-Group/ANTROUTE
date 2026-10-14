/**
 * SEAM: the CSV upload panel.
 *
 * The input features are not finalized, so the panel deliberately does NOT parse the file.
 * It validates the extension and size, then hands the raw File to onFileAccepted. When the
 * schema is settled, parse inside that callback -- UploadFilePanel itself should not need
 * to change.
 */

export interface UploadedDataset {
  /** The raw handle. Nothing has read its contents. */
  file: File;
  name: string;
  sizeBytes: number;
  lastModified: number;
}

export type UploadRejection =
  | { code: 'unsupported-type'; file: File; accept: string[] }
  | { code: 'too-large'; file: File; maxSizeBytes: number }
  | { code: 'empty'; file: File };

export type UploadStatus = 'idle' | 'validating' | 'ready' | 'error';

export interface UploadFilePanelProps {
  /** Extensions, with the dot. Defaults to ['.csv']. */
  accept?: string[];
  /** Defaults to 10 MB. */
  maxSizeBytes?: number;
  disabled?: boolean;
  /** THE SEAM: the only call a data branch has to implement. */
  onFileAccepted: (dataset: UploadedDataset) => void | Promise<void>;
  onFileRejected?: (rejection: UploadRejection) => void;
  onClear?: () => void;
  /** Caller-driven, so parse or upload progress can be shown later without touching
   *  the panel. */
  status?: UploadStatus;
  statusMessage?: string | null;
}
