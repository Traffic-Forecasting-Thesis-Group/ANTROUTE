import { useCallback, useRef, useState } from 'react';
import { FileText, UploadCloud, X } from 'lucide-react';

import { cn } from '@/lib/cn';
import type {
  UploadFilePanelProps,
  UploadRejection,
  UploadedDataset,
} from '@/types/upload';

const DEFAULT_ACCEPT = ['.csv'];
const DEFAULT_MAX_BYTES = 10 * 1024 * 1024;

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function hasAcceptedExtension(file: File, accept: string[]): boolean {
  const lower = file.name.toLowerCase();
  return accept.some((ext) => lower.endsWith(ext.toLowerCase()));
}

/**
 * The upload box from the dashboard mockup.
 *
 * It deliberately does NOT read the file: the input features are not finalized, so there
 * is no schema to parse against and no backend endpoint to send to. It validates the
 * extension and size, then hands the raw File to onFileAccepted -- that callback is the
 * seam (see src/types/upload.ts). Parsing belongs there, not here.
 */
export default function UploadFilePanel({
  accept = DEFAULT_ACCEPT,
  maxSizeBytes = DEFAULT_MAX_BYTES,
  disabled = false,
  onFileAccepted,
  onFileRejected,
  onClear,
  status = 'idle',
  statusMessage = null,
}: UploadFilePanelProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [selected, setSelected] = useState<UploadedDataset | null>(null);
  const [localError, setLocalError] = useState<string | null>(null);

  const reject = useCallback(
    (rejection: UploadRejection, message: string) => {
      setSelected(null);
      setLocalError(message);
      onFileRejected?.(rejection);
    },
    [onFileRejected]
  );

  const handleFile = useCallback(
    (file: File) => {
      setLocalError(null);

      if (!hasAcceptedExtension(file, accept)) {
        reject(
          { code: 'unsupported-type', file, accept },
          `Only ${accept.join(', ')} files are supported.`
        );
        return;
      }
      if (file.size === 0) {
        reject({ code: 'empty', file }, 'That file is empty.');
        return;
      }
      if (file.size > maxSizeBytes) {
        reject(
          { code: 'too-large', file, maxSizeBytes },
          `That file is larger than ${formatSize(maxSizeBytes)}.`
        );
        return;
      }

      const dataset: UploadedDataset = {
        file,
        name: file.name,
        sizeBytes: file.size,
        lastModified: file.lastModified,
      };
      setSelected(dataset);
      void onFileAccepted(dataset);
    },
    [accept, maxSizeBytes, onFileAccepted, reject]
  );

  const clear = () => {
    setSelected(null);
    setLocalError(null);
    if (inputRef.current) inputRef.current.value = '';
    onClear?.();
  };

  const message = localError ?? statusMessage;
  const isError = Boolean(localError) || status === 'error';

  return (
    <section>
      <h2 className="mb-2 text-sm font-semibold text-slate-700">Upload File</h2>

      <input
        ref={inputRef}
        type="file"
        accept={accept.join(',')}
        disabled={disabled}
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) handleFile(file);
        }}
        className="sr-only"
      />

      {selected ? (
        <div className="flex items-center gap-2 rounded-lg border border-brand bg-brand-light px-3 py-2.5">
          <FileText size={16} className="shrink-0 text-brand" aria-hidden />
          <span className="min-w-0 flex-1">
            <span className="block truncate text-xs font-medium text-slate-800">
              {selected.name}
            </span>
            <span className="block text-[11px] text-slate-500">
              {formatSize(selected.sizeBytes)}
              {status === 'validating' && ' · checking…'}
            </span>
          </span>
          <button
            type="button"
            onClick={clear}
            aria-label="Remove file"
            className="shrink-0 text-slate-400 hover:text-red-500"
          >
            <X size={15} />
          </button>
        </div>
      ) : (
        <button
          type="button"
          disabled={disabled}
          onClick={() => inputRef.current?.click()}
          onDragOver={(e) => {
            e.preventDefault();
            if (!disabled) setIsDragging(true);
          }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setIsDragging(false);
            if (disabled) return;
            const file = e.dataTransfer.files?.[0];
            if (file) handleFile(file);
          }}
          className={cn(
            'flex w-full flex-col items-center gap-1 rounded-lg border-2 border-dashed px-4 py-6 transition-colors',
            isDragging
              ? 'border-brand bg-brand-light'
              : 'border-slate-300 bg-white hover:border-brand hover:bg-brand-light/50',
            disabled && 'cursor-not-allowed opacity-60'
          )}
        >
          <UploadCloud size={22} className="text-brand" aria-hidden />
          <span className="text-xs font-medium text-slate-700">
            Choose a file or drag and drop it here
          </span>
          <span className="text-[11px] text-slate-400">
            File supported: {accept.join(', ')}
          </span>
        </button>
      )}

      {message && (
        <p className={cn('mt-1.5 text-xs', isError ? 'font-medium text-red-600' : 'text-slate-500')}>
          {message}
        </p>
      )}
    </section>
  );
}
