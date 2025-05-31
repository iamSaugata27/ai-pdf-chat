import { readFileSync } from 'fs';
import { createHash } from 'crypto';

export const getPdfHash = (filePath: string): string => {
    const buffer = readFileSync(filePath);
    return createHash('sha256').update(buffer).digest('hex');
}