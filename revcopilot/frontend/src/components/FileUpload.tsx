'use client';

import { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, Binary, FileWarning } from 'lucide-react';
import { toast } from 'react-hot-toast';

interface FileUploadProps {
  onFileUpload: (file: File) => Promise<void>;
  isUploading: boolean;
}

export default function FileUpload({ onFileUpload, isUploading }: FileUploadProps) {
  const [dragOver, setDragOver] = useState(false);

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (!file) return;

    // Check file size (limit to 50MB)
    if (file.size > 50 * 1024 * 1024) {
      toast.error('File too large. Maximum size is 50MB.');
      return;
    }

    // Check if it's likely a binary
    const isBinary = !file.type.startsWith('text/') && !file.type.includes('json');
    if (!isBinary) {
      toast.error('Please upload a binary executable file.');
      return;
    }

    await onFileUpload(file);
  }, [onFileUpload]);

  const { getRootProps, getInputProps, isDragActive, open } = useDropzone({
    onDrop,
    maxFiles: 1,
    disabled: isUploading,
    noClick: true,
    noKeyboard: true,
  });

  return (
    <div
      {...getRootProps({
        onClick: () => {
          if (!isUploading) {
            open();
          }
        },
      })}
      className={`
        relative border-4 border-dashed rounded-3xl p-12 text-center cursor-pointer
        transition-all duration-300 transform hover:scale-[1.02]
        ${isDragActive ? 'border-blue-500 bg-blue-50 scale-[1.02]' : 'border-gray-300'}
        ${dragOver ? 'border-blue-500 bg-blue-50' : ''}
        ${isUploading ? 'opacity-50 cursor-not-allowed' : ''}
      `}
      onMouseEnter={() => setDragOver(true)}
      onMouseLeave={() => setDragOver(false)}
    >
      <input {...getInputProps()} />
      
      <div className="space-y-6">
        <div className="flex justify-center">
          <div className={`
            p-4 rounded-2xl transition-all duration-300
            ${isDragActive ? 'bg-blue-100 scale-110' : 'bg-gray-100'}
          `}>
            {isUploading ? (
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
            ) : (
              <Binary className="h-12 w-12 text-blue-600" />
            )}
          </div>
        </div>
        
        <div className="space-y-2">
          <h2 className="text-2xl font-bold text-gray-800">
            {isUploading ? 'Analyzing Binary...' : 
             isDragActive ? 'Drop the binary here' : 'Drag & drop a binary'}
          </h2>
          
          <p className="text-gray-600">
            {isUploading 
              ? 'Running static analysis and symbolic execution...' 
              : 'or click to browse for executable files'}
          </p>
          
          <div className="flex items-center justify-center gap-2 text-sm text-gray-500">
            <FileWarning className="h-4 w-4" />
            <span>Supports ELF, PE, Mach-O formats</span>
          </div>
        </div>
        
        {!isUploading && !isDragActive && (
          <div className="pt-4">
            <button
              type="button"
              onClick={(event) => {
                event.preventDefault();
                event.stopPropagation();
                open();
              }}
              className="inline-flex items-center gap-2 bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 px-6 rounded-xl transition-colors"
            >
              <Upload className="h-5 w-5" />
              Choose File
            </button>
          </div>
        )}
      </div>
    </div>
  );
}